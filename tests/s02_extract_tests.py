"""S2 tests: the shell wrapper (syntax, dry-run, full run on a synthetic raw tree through the real extractor), S2c segments
rebuild and S2d acceptance. Everything runs under $D/tmp (pytest basetemp honours TMPDIR from env.sh); no real identifier
appears anywhere (tests/synth_qlog fake ids)."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import synth_qlog as SQ
from vtt import config as C
from vtt import extract_accept as EA
from vtt import reuse as RU
from vtt import segments as SG
from vtt.private import HEX16, ROUTE_NUM
from vtt.tocf_bridge import tocf

R = Path(__file__).resolve().parents[1]
SHELL = R / "scripts" / "s02_extract_all.sh"
TSCHEMA = tocf("schema")


def load_vtt_script(name: str):
    spec = importlib.util.spec_from_file_location(name, R / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# route name -> (car_dir, dongle n, route n, kwargs)
ROUTES = {
    "A": ("CAR_A", 1, 1, dict(profile="ice", version="0.9.7-release", n_seg=3, last_seg_s=20.0, unknown_n=5, truncate_seg=1, zst_segs=(0,))),
    "B": ("CAR_B", 2, 2, dict(profile="bev", version="2026.003.000", n_seg=2,
                           topics=("carState", "controlsState", "carControl", "selfdriveState", "livePose"))),   # radar silently gone
    "C": ("CAR_C", 3, 3, dict(profile="hev", version="0.10.1", n_seg=2)),
    "D": ("CAR_A", 4, 4, dict(profile="ice", version="0.8.13", n_seg=1)),
}
MISSING = ("CAR_Z", 9, 9)      # in uid_map and in a chunk, absent from the raw tree


def _env(root: Path) -> dict:
    env = dict(os.environ)
    env.update(VTT_DATA_ROOT=str(root), VTT_RAW_ROOT=str(root / "raw"), D=str(root))
    return env


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("s02")
    raw, priv = root / "raw", root / "private"
    rows, uids = [], {}
    for name, (car, dn, rn, kw) in ROUTES.items():
        dongle, route = SQ.fake_dongle(dn), SQ.fake_route(rn)
        SQ.write_route(raw / car / dongle / route, dongle, car, **kw)
        rows.append(dict(car_dir=car, dongle_dir=dongle, dongle_log=dongle, route_id=route))
    car, dn, rn = MISSING
    rows.append(dict(car_dir=car, dongle_dir=SQ.fake_dongle(dn), dongle_log=SQ.fake_dongle(dn), route_id=SQ.fake_route(rn)))
    um = SQ.write_private(priv, rows)
    for name, r in zip(list(ROUTES) + ["MISSING"], um.itertuples(index=False)):
        uids[name] = r.route_uid
    chunks = root / "build" / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "routes_chunk_00.csv").write_text("route_uid\n" + "\n".join(uids[n] for n in ("A", "B", "MISSING")) + "\n")
    (chunks / "routes_chunk_long.csv").write_text("route_uid\n" + "\n".join(uids[n] for n in ("C", "D")) + "\n")
    return {"root": root, "raw": raw, "priv": priv, "uids": uids, "um": um}


@pytest.fixture(scope="module")
def full_run(tree) -> dict:
    """The shell wrapper end to end (real extractor, s02c, s02d) on the synthetic tree."""
    root = tree["root"]
    p = subprocess.run(["bash", str(SHELL), "--data-root", str(root), "--raw-root", str(tree["raw"])], capture_output=True, text=True,
                       env=_env(root), timeout=600)
    return {"rc": p.returncode, "out": p.stdout + p.stderr, **tree}


# ----------------------------------------------------------------------------- shell wrapper
def test_shell_syntax():
    assert subprocess.run(["bash", "-n", str(SHELL)], capture_output=True).returncode == 0


def test_shell_dry_run_prints_commands_only(tree, tmp_path):
    root = tmp_path / "dry"                                  # its own root: pytest may run the full-run fixture first
    (root / "build" / "chunks").mkdir(parents=True)
    for f in ("routes_chunk_00.csv", "routes_chunk_long.csv"):
        (root / "build" / "chunks" / f).write_text((tree["root"] / "build" / "chunks" / f).read_text())
    p = subprocess.run(["bash", str(SHELL), "--data-root", str(root), "--raw-root", str(tree["raw"]), "--dry-run"],
                       capture_output=True, text=True, env=_env(root), timeout=120)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out
    assert str(C.EXTRACTOR_SCRIPT) in out and "--no-frame-count" in out
    cmds = [ln for ln in out.splitlines() if "s02_extract_canonical.py" in ln and ln.startswith("+ ")]
    assert len(cmds) == 2 and "routes_chunk_00.csv" in cmds[0] and "routes_chunk_long.csv" in cmds[1]
    assert f"--workers {C.EXTRACT_WORKERS} " in cmds[0] + " " and f"--workers {C.EXTRACT_WORKERS_LONG} " in cmds[1] + " "
    assert "nice -n 15 ionice -c3" in cmds[0] and "--private-root" in cmds[0] and str(root / "work") in cmds[0]
    assert "s02c_segments_table.py" in out and "s02d_accept.py" in out and "cp -al" in out
    assert not (root / "work" / "signals").exists()           # nothing was run


def test_shell_full_run_extracts_and_judges(full_run):
    root, uids = full_run["root"], full_run["uids"]
    out = full_run["out"]
    assert full_run["rc"] == 2, out                           # the radar-less 2026 route and the missing route reject the build
    sig = root / "work" / "signals"
    assert sorted(d.name for d in sig.iterdir() if not d.name.startswith(".")) == sorted(uids[n] for n in ROUTES)
    assert (root / "build" / "logs" / "s02_routes_chunk_00.log").exists() and (root / "build" / "logs" / "s02_routes_chunk_long.log").exists()
    fails = pd.read_csv(root / "work" / "logs" / "s02_failures.csv")
    assert len(fails) == 1 and fails.route_uid.iloc[0] == uids["MISSING"]
    md = (root / "build" / "qc" / "s02_accept.md").read_text()
    assert "REJECTED" in md and "era 2026" in md and "missing" in md
    assert (root / "build" / "qc" / "s02_accept.csv").exists()
    assert "nothing reused" in out                            # the test salt differs from the seed salt: no hardlink reuse
    # re-running is safe: the extractor reports everything done and the verdict is unchanged
    p = subprocess.run(["bash", str(SHELL), "--data-root", str(root), "--raw-root", str(full_run["raw"]), "--status"],
                       capture_output=True, text=True, env=_env(root), timeout=120)
    assert p.returncode == 0 and "already done 2" in p.stdout and "to extract 0" in p.stdout, p.stdout + p.stderr


def test_no_identifier_in_outputs(full_run):
    root = full_run["root"]
    texts = [full_run["out"], (root / "build" / "qc" / "s02_accept.md").read_text(), (root / "build" / "qc" / "s02_accept.csv").read_text()]
    texts += [p.read_text() for p in (root / "build" / "logs").glob("*.log")]
    for t in texts:
        t = re.sub(r"\d+\.\d+", "<dec>", t)                   # decimals first, as the release gate does
        assert not HEX16.search(t) and not ROUTE_NUM.search(t), "identifier-like token in a public-side output"


# ----------------------------------------------------------------------------- S2c
def test_segments_table_contract_and_rates(full_run):
    root, uids = full_run["root"], full_run["uids"]
    seg = pd.read_parquet(root / "work" / "tables" / "segments.parquet")
    assert list(seg.columns) == list(TSCHEMA.SEGMENTS)
    for c, dt in TSCHEMA.SEGMENTS.items():
        assert str(seg[c].dtype) == ("object" if dt == "str" else dt), (c, seg[c].dtype)
    a = seg[seg.route_uid == uids["A"]].set_index("seg")
    assert a.src_kind.tolist() == ["qlog"] * 3 and a.at[0, "unknown_union_n"] == 5 and a.at[1, "unknown_union_n"] == 0
    assert abs(a.at[0, "rate_carState"] - 10) < 0.05 and abs(a.at[0, "rate_radarState"] - 4) < 0.05 and np.isnan(a.at[0, "rate_dm"])
    b = seg[seg.route_uid == uids["B"]]
    assert b.rate_radarState.isna().all() and (abs(b.rate_carState - 10) < 0.05).all()
    counts = pd.read_parquet(root / "work" / "tables" / SG.SEGMENT_COUNTS_FILE)
    assert list(counts.columns) == SG.SEGMENT_COUNTS_COLUMNS
    ca = counts[counts.route_uid == uids["A"]].set_index("seg")
    assert ca.n_carState.tolist() == [600, 360, 200] and ca.n_radarState.tolist() == [240, 144, 80] and ca.n_pose.tolist() == [300, 180, 100]
    assert ca.span_s.iloc[0] > 55 and ca.span_s.iloc[2] < 55
    rm = pd.read_parquet(root / "work" / "tables" / SG.ROUTES_META_FILE).set_index("route_uid")
    assert list(rm.reset_index().columns) == SG.ROUTES_META_COLUMNS
    assert rm.at[uids["A"], "software_era"] == "0.9" and rm.at[uids["B"], "software_era"] == "2026"
    assert rm.at[uids["C"], "software_era"] == "0.10" and rm.at[uids["D"], "software_era"] == "0.8"
    assert rm.at[uids["A"], "n_truncated"] == 1 and rm.at[uids["A"], "n_seg_qlog"] == 3 and rm.at[uids["B"], "n_truncated"] == 0
    assert (rm.extractor_version == C.EXTRACTOR_VERSION).all() and (rm.pose_source == "livePose").all()


def test_segments_rebuild_matches_extractor_rates(full_run):
    """The rebuilt rates equal what the extractor itself wrote into its own segments.parquet (same rows, same formula)."""
    root = full_run["root"]
    mine = pd.read_parquet(root / "work" / "tables" / "segments.parquet").set_index(["route_uid", "seg"])
    res = SG.rebuild(root / "work" / "signals", workers=1)
    again = res["segments"].set_index(["route_uid", "seg"])
    assert len(again) == len(mine) == 8 and not res["errors"]
    for col in ("rate_carState", "rate_radarState", "t0_mono_ns", "t1_mono_ns", "unknown_union_n"):
        pd.testing.assert_series_equal(mine[col], again[col], check_names=False)


def test_software_era_fallback():
    f = SG.software_era_fallback
    assert f("0.8.13") == "0.8" and f("0.9.7-release") == "0.9" and f("0.10.1") == "0.10" and f("0.11.2-sunnypilot") == "0.11"
    assert f("2025.12.3") == "2025" and f("2026.003.000") == "2026"
    assert f("0.7.10") == "other" and f(None) == "other" and f("garbage") == "other" and f("2024.1.0") == "other"
    for v in ("0.8.1", "0.9.9", "2026.1.1", "x"):
        assert SG.software_era(v) in C.SOFTWARE_ERAS


def test_s02c_script_subset_and_dry_run(full_run, tmp_path):
    root, uids = full_run["root"], full_run["uids"]
    s02c = load_vtt_script("s02c_segments_table")
    routes = tmp_path / "subset.csv"
    routes.write_text("route_uid\n" + "\n".join(uids[n] for n in ("A", "C", "D")) + "\n")
    out = tmp_path / "tables"
    assert s02c.main(["--data-root", str(root), "--routes", str(routes), "--tables-dir", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert s02c.main(["--data-root", str(root), "--routes", str(routes), "--tables-dir", str(out), "--workers", "2"]) == 0
    rm = pd.read_parquet(out / SG.ROUTES_META_FILE)
    assert set(rm.route_uid) == {uids[n] for n in ("A", "C", "D")}
    assert len(pd.read_parquet(out / "segments.parquet")) == 6


# ----------------------------------------------------------------------------- S2d
def test_accept_passes_without_the_bad_route(full_run, tmp_path):
    root, uids = full_run["root"], full_run["uids"]
    s02c, s02d = load_vtt_script("s02c_segments_table"), load_vtt_script("s02d_accept")
    routes = tmp_path / "good.csv"
    routes.write_text("route_uid\n" + "\n".join(uids[n] for n in ("A", "C", "D")) + "\n")
    tables, chunks, qc = tmp_path / "tables", tmp_path / "chunks", tmp_path / "qc"
    chunks.mkdir()
    (chunks / "routes_chunk_00.csv").write_text(routes.read_text())
    assert s02c.main(["--data-root", str(root), "--routes", str(routes), "--tables-dir", str(tables), "--workers", "1"]) == 0
    # the signals dir still holds route B: point the acceptance at a signals view without it (stale check must not fire)
    view = tmp_path / "signals"
    view.mkdir()
    for n in ("A", "C", "D"):
        os.symlink(root / "work" / "signals" / uids[n], view / uids[n])
    rc = s02d.main(["--data-root", str(root), "--tables-dir", str(tables), "--signals-dir", str(view), "--chunks-dir", str(chunks),
                    "--uid-map", str(root / "private" / "uid_map.parquet"), "--failures", str(tmp_path / "none.csv"), "--out-dir", str(qc)])
    assert rc == 0
    era = pd.read_csv(qc / "s02_accept.csv").set_index("software_era")
    assert era.at["0.9", "n_full"] == 1 and era.at["0.9", "good_frac"] == 1.0 and era.at["0.9", "n_short"] == 2
    assert era.at["0.10", "n_good"] == 2 and era.at["ALL", "n_good"] == 4 and era.at["ALL", "unk_1_10"] == 1
    assert "ACCEPTED" in (qc / "s02_accept.md").read_text()
    # the same tables judged against the full signals dir: stale (route B extracted but not in the tables) -> rejected
    rc = s02d.main(["--data-root", str(root), "--tables-dir", str(tables), "--chunks-dir", str(chunks), "--out-dir", str(qc)])
    assert rc == 2 and "stale" in (qc / "s02_accept.md").read_text()


def test_era_table_and_evaluate_rules():
    rm = pd.DataFrame([dict(route_uid="r1", software_era="0.9", extractor_version=C.EXTRACTOR_VERSION, n_truncated=0, n_read_error=0),
                       dict(route_uid="r2", software_era="0.9", extractor_version="s02-v2", n_truncated=1, n_read_error=0)])
    counts = pd.DataFrame([dict(route_uid="r1", seg=0, src_kind="qlog", span_s=59.9, n_carState=600, n_radarState=240),
                           dict(route_uid="r1", seg=1, src_kind="qlog", span_s=59.9, n_carState=600, n_radarState=120),   # low radar (3 Hz writer): reported, still good
                           dict(route_uid="r1", seg=2, src_kind="qlog", span_s=59.9, n_carState=600, n_radarState=10),    # radar absent: not good
                           dict(route_uid="r2", seg=0, src_kind="qlog", span_s=30.0, n_carState=300, n_radarState=120),   # short: not judged
                           dict(route_uid="r2", seg=1, src_kind="rlog_dec", span_s=59.9, n_carState=600, n_radarState=240)])
    segs = pd.DataFrame([dict(route_uid="r1", seg=0, unknown_union_n=0), dict(route_uid="r1", seg=1, unknown_union_n=320),
                         dict(route_uid="r1", seg=2, unknown_union_n=0),
                         dict(route_uid="r2", seg=0, unknown_union_n=50), dict(route_uid="r2", seg=1, unknown_union_n=0)])
    era = EA.era_table(segs, counts, rm).set_index("software_era")
    assert era.at["0.9", "n_full"] == 3 and era.at["0.9", "n_good"] == 2 and era.at["0.9", "n_low_radar"] == 2 and era.at["0.9", "n_radar_absent"] == 1 and era.at["0.9", "n_short"] == 1
    assert era.at["0.9", "n_rlog_dec"] == 1 and era.at["0.9", "unk_301p"] == 1 and era.at["0.9", "unk_11_100"] == 1 and era.at["0.9", "unk_0"] == 2
    assert era.at["0.9", "n_version_mismatch"] == 1 and era.at["ALL", "n_truncated"] == 1
    ok, reasons = EA.evaluate(era.reset_index(), n_expected=200, n_missing=0)
    assert not ok and any("era 0.9" in r for r in reasons) and any("extractor_version" in r for r in reasons)
    ok, reasons = EA.evaluate(era.reset_index().assign(n_version_mismatch=0, good_frac=1.0), n_expected=200, n_missing=1)
    assert ok and reasons == []                                                    # 0.5 % missing is the limit, not a breach
    ok, reasons = EA.evaluate(era.reset_index().assign(n_version_mismatch=0, good_frac=1.0), n_expected=200, n_missing=2)
    assert not ok and reasons == ["routes missing from work/signals (failed or not extracted): 2/200 = 0.0100 > 0.005"]
    ok, reasons = EA.evaluate(era.reset_index().assign(n_version_mismatch=0, good_frac=1.0), n_expected=1000, n_missing=5)
    assert ok and reasons == []
    ok, reasons = EA.evaluate(era.reset_index().assign(n_version_mismatch=0, good_frac=1.0), n_expected=1000, n_missing=6)
    assert not ok


def test_failures_and_expected_sources(tmp_path):
    (tmp_path / "f.csv").write_text("route_uid,status,error,when\nu1,error:FileNotFoundError,x,2026\nu2,error:ValueError,y,2026\nu1,error:FileNotFoundError,x,2026\n")
    f = EA.failures_table(tmp_path / "f.csv", extracted={"u2"})
    assert f == {"n_rows": 3, "n_uids": 2, "n_unresolved": 1, "status_hist": {"error:FileNotFoundError": 2, "error:ValueError": 1}}
    assert EA.failures_table(tmp_path / "missing.csv", set())["n_rows"] == 0
    uids, src = EA.expected_uids(tmp_path / "nochunks", tmp_path / "no_uid_map.parquet")
    assert uids == set() and src == "none"
    pd.DataFrame({"route_uid": ["a", "b"]}).to_parquet(tmp_path / "um.parquet", index=False)
    assert EA.expected_uids(tmp_path / "nochunks", tmp_path / "um.parquet") == ({"a", "b"}, "uid_map")
    ch = tmp_path / "chunks"
    ch.mkdir()
    (ch / "routes_chunk_00.csv").write_text("route_uid\na\n")
    (ch / "routes_chunk_long.csv").write_text("route_uid\nc\n")
    assert EA.expected_uids(ch, tmp_path / "um.parquet") == ({"a", "c"}, "2 chunk files")


# ----------------------------------------------------------------------------- reuse verification
def _raise_exdev(*_a, **_k):
    raise OSError(18, "Invalid cross-device link")


def test_reuse_verify_sizes_fresh_and_fill(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for u in ("u1", "u2", "u3", "u4"):
        (src / u).mkdir(parents=True)
        for f in ("carState.parquet", "route_meta.json"):
            (src / u / f).write_text(f"{u}-{f}-payload")
    old = time.time() - 3600.0
    for u in ("u1", "u2", "u3", "u4"):
        os.utime(src / u / "route_meta.json", (old, old))              # the source extraction is older than this build
    (src / "u5").mkdir()
    (src / "u5" / "carState.parquet").write_text("no route_meta.json: not reusable")
    (dst / "u1").mkdir(parents=True)
    for f in ("carState.parquet", "route_meta.json"):
        os.link(src / "u1" / f, dst / "u1" / f)                      # complete, hardlinked
    (dst / "u2").mkdir()
    for f in ("carState.parquet", "route_meta.json"):
        shutil.copy2(src / "u2" / f, dst / "u2" / f)
    (dst / "u2" / "carState.parquet").write_text("u2")               # same names, cut copy -> incomplete by size
    (dst / "u4").mkdir()
    (dst / "u4" / "route_meta.json").write_text("re-extracted by this build")   # newer route_meta.json -> fresh, kept as is
    (dst / "u9").mkdir()
    (dst / "u9" / "route_meta.json").write_text("not a source route")           # ignored
    (dst / ".u3.building").mkdir()
    res = RU.verify(src, dst)
    keys = ("source_dirs", "source_usable", "dest_dirs", "complete", "fresh", "incomplete", "removed", "filled", "fill_failed", "missing", "hardlinked")
    assert {k: res[k] for k in keys} == {"source_dirs": 5, "source_usable": 4, "dest_dirs": 4, "complete": 1, "fresh": 1, "incomplete": 1,
                                         "removed": 0, "filled": 0, "fill_failed": 0, "missing": 2, "hardlinked": 1}
    assert RU.main(["--src", str(src), "--dst", str(dst)]) == 2 and (dst / "u2").exists()
    # --fill alone rebuilds the absent route but cannot rebuild over the kept incomplete one
    res = RU.verify(src, dst, fill=True)
    assert res["filled"] == 1 and res["fill_failed"] == 1 and res["missing"] == 1 and RU._files(src / "u3") == RU._files(dst / "u3")
    assert res["hardlinked"] == 2 and res["filled_linked_files"] == 2 and res["filled_copied_files"] == 0
    shutil.rmtree(dst / "u3")
    # a link failure (EXDEV) falls back to a copy per file; --fix --fill leaves every source route complete -> exit 0
    monkeypatch.setattr(RU.os, "link", _raise_exdev)
    assert RU.main(["--src", str(src), "--dst", str(dst), "--fix", "--fill"]) == 0
    res = RU.verify(src, dst)
    assert res["complete"] == 3 and res["fresh"] == 1 and res["incomplete"] == 0 and res["missing"] == 0 and res["hardlinked"] == 1
    for u in ("u2", "u3"):
        assert RU._files(src / u) == RU._files(dst / u) and os.stat(src / u / "route_meta.json").st_ino != os.stat(dst / u / "route_meta.json").st_ino
    assert (dst / "u4" / "route_meta.json").read_text() == "re-extracted by this build" and (dst / "u9").exists()
    assert not any(p.name.endswith(".reusing") for p in dst.iterdir())
    assert RU.verify(tmp_path / "nowhere", dst)["source_dirs"] == 0 and RU.verify(src, tmp_path / "nodst")["missing"] == 4


# ----------------------------------------------------------------------------- the shell's reuse step and pre-flight (env overrides)
def _second_root(full_run, root: Path, s01_accepted=None) -> dict:
    """A data root with the same private salt / uid_map and chunk lists as the full run (same route_uids), whose reuse source and
    seed root are the full run's work/signals and private dir (env overrides read by vtt.config)."""
    rows = full_run["um"][["car_dir", "dongle_dir", "dongle_log", "route_id"]].to_dict("records")
    SQ.write_private(root / "private", rows)
    (root / "build" / "chunks").mkdir(parents=True)
    for f in (full_run["root"] / "build" / "chunks").glob("*.csv"):
        (root / "build" / "chunks" / f.name).write_text(f.read_text())
    if s01_accepted is not None:
        (root / "build" / "logs").mkdir(parents=True, exist_ok=True)
        (root / "build" / "logs" / "s01_summary.json").write_text(json.dumps({"stage": "S1", "accepted": s01_accepted,
                                                                                "problems": [] if s01_accepted else ["chunk mismatch"]}))
    env = _env(root)
    env.update(VTT_RAW_ROOT=str(full_run["raw"]), VTT_REUSE_EXISTING_SIGNALS=str(full_run["root"] / "work" / "signals"),
               VTT_SEED_PRIVATE_ROOT=str(full_run["root"] / "private"))
    return env


def _sh(root: Path, env: dict, *args, timeout=600) -> tuple[int, str]:
    p = subprocess.run(["bash", str(SHELL), "--data-root", str(root), "--raw-root", env["VTT_RAW_ROOT"], *args], capture_output=True,
                       text=True, env=env, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def _route_dirs(d: Path) -> list[str]:
    return sorted(x.name for x in d.iterdir() if x.is_dir() and not x.name.startswith("."))


def test_shell_status_is_read_only_and_reuse_links_the_earlier_extraction(full_run, tmp_path):
    root = tmp_path / "reuse"
    env = _second_root(full_run, root, s01_accepted=True)
    src, sig, marker = full_run["root"] / "work" / "signals", root / "work" / "signals", root / "work" / ".reuse_signals_done"
    rc, out = _sh(root, env, "--status", timeout=180)                 # before anything ran: reports, creates nothing
    assert rc == 0, out
    assert "reuse: pending" in out and out.count("already done 0") == 2 and not sig.exists() and not marker.exists()
    rc, out = _sh(root, env)                                           # real run: the four extracted routes are reused
    assert rc == 2, out                                                # same verdict as the first build (radar-less 2026 route, missing route)
    assert "S1 summary: accepted" in out and ("mode=hardlink" in out or "mode=copy" in out) and "marker written" in out
    assert marker.exists() and "missing=0" in marker.read_text() and "complete=4" in marker.read_text()
    res = RU.verify(src, sig)
    assert res["complete"] == 4 and res["missing"] == 0 and res["incomplete"] == 0 and res["dest_dirs"] == 4
    assert _route_dirs(sig) == _route_dirs(src)
    for u in _route_dirs(src):
        assert RU._files(src / u) == RU._files(sig / u)
    if "mode=hardlink" in out:
        assert res["hardlinked"] == 4
    assert out.count("already done 2") == 2 and "to extract 1 " in out and "to extract 0 " in out   # A, B + missing; C, D
    assert len(pd.read_csv(root / "work" / "logs" / "s02_failures.csv")) == 1
    assert "REJECTED" in (root / "build" / "qc" / "s02_accept.md").read_text()
    before = marker.read_text()
    rc, out = _sh(root, env, "--status", timeout=180)                 # the marker is reported, nothing changes
    assert rc == 0 and "reuse: already done" in out and marker.read_text() == before and _route_dirs(sig) == _route_dirs(src)


def test_shell_reuse_marker_waits_for_an_unreadable_source_route(full_run, tmp_path):
    """cp fails on an unreadable source dir -> no marker (next run retries); once readable, the next run fills that one dir
    without a bulk cp and writes the marker."""
    root, src = tmp_path / "reuse2", tmp_path / "src"
    env = _second_root(full_run, root)
    env["VTT_REUSE_EXISTING_SIGNALS"] = str(src)
    src.mkdir()
    for u in _route_dirs(full_run["root"] / "work" / "signals"):
        shutil.copytree(full_run["root"] / "work" / "signals" / u, src / u)
    extra = src / ("e" * 16)                                           # a fifth source route (not in uid_map), unreadable at first
    shutil.copytree(src / _route_dirs(src)[0], extra)
    os.chmod(extra, 0)
    marker = root / "work" / ".reuse_signals_done"
    try:
        rc, out = _sh(root, env, "--only-chunk", "routes_chunk_long")
        assert rc == 2, out
        assert "marker NOT written" in out and not marker.exists() and "complete=4" in out and "missing=1" in out
        assert _route_dirs(root / "work" / "signals") == sorted(u for u in _route_dirs(src) if u != extra.name)
        assert "already done 2" in out and "to extract 0 " in out
    finally:
        os.chmod(extra, 0o755)
    rc, out = _sh(root, env, "--only-chunk", "routes_chunk_long")
    assert rc == 2, out
    assert "bulk cp skipped" in out and "filled=1" in out and "missing=0" in out and "marker written" in out and marker.exists()
    assert RU.verify(src, root / "work" / "signals")["complete"] == 5
    assert not any(p.name.endswith(".reusing") for p in (root / "work" / "signals").iterdir())


def test_shell_refuses_when_s1_not_accepted(full_run, tmp_path):
    root = tmp_path / "gate"
    env = _second_root(full_run, root, s01_accepted=False)
    rc, out = _sh(root, env, timeout=180)
    assert rc == 1 and "accepted=false" in out and not (root / "work" / "signals").exists(), out
    rc, out = _sh(root, env, "--status", timeout=180)                 # status is informational: no gate, nothing created
    assert rc == 0 and "accepted=false" not in out and not (root / "work").exists(), out
    rc, out = _sh(root, env, "--force", "--skip-reuse", "--only-chunk", "routes_chunk_long")
    assert rc == 2, out
    assert "continuing (--force)" in out and "reuse: skipped" in out
    assert _route_dirs(root / "work" / "signals") == sorted(full_run["uids"][n] for n in ("C", "D"))


def test_software_era_out_of_list_is_other_and_every_present_era_is_judged():
    assert SG.software_era("2027.1.0") == "other" == SG.software_era_fallback("2027.1.0")
    assert SG.software_era("1.2.3") == "other" and SG.software_era("2026.3.1") == "2026" and SG.software_era("0.9.2") == "0.9"
    rm = pd.DataFrame([dict(route_uid="r1", software_era="2027", extractor_version=C.EXTRACTOR_VERSION, n_truncated=0, n_read_error=0),
                       dict(route_uid="r2", software_era=None, extractor_version=C.EXTRACTOR_VERSION, n_truncated=0, n_read_error=0)])
    counts = pd.DataFrame([dict(route_uid="r1", seg=0, src_kind="qlog", span_s=59.9, n_carState=100, n_radarState=0),
                           dict(route_uid="r2", seg=0, src_kind="qlog", span_s=59.9, n_carState=600, n_radarState=240)])
    segs = pd.DataFrame([dict(route_uid="r1", seg=0, unknown_union_n=0), dict(route_uid="r2", seg=0, unknown_union_n=0)])
    assert EA.eras_present(rm) == ["other", "2027"]
    era = EA.era_table(segs, counts, rm).set_index("software_era")
    assert list(era.index) == ["other", "2027", "ALL"]
    assert era.at["2027", "n_full"] == 1 and era.at["2027", "good_frac"] == 0.0 and era.at["other", "good_frac"] == 1.0
    ok, reasons = EA.evaluate(era.reset_index(), n_expected=2, n_missing=0)
    assert not ok and any("era 2027" in r for r in reasons)
