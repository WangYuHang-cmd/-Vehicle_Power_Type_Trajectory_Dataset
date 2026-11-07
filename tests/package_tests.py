"""Release assembly tests: layout, hardlinks, shard contents and split assignment, idempotency, dry run, acceptance,
CHECKSUMS coverage and the two CLI scripts."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tests.synth_build import make_build
from vtt import config as C
from vtt import schema as S
from vtt.gate import scan as SC
from vtt.package import AcceptanceError, Roots, package, write_checksums
from vtt.paths import P

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site():
    P.TMP.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_pkg_", dir=P.TMP))
    info = make_build(root / "build")
    roots = Roots.from_data_root(root)
    summary = package(roots, per_shard=4, workers=2)
    yield root, roots, info, summary
    if not os.environ.get("VTT_KEEP_TMP"):
        shutil.rmtree(root, ignore_errors=True)


def test_summary_and_layout(site):
    root, roots, info, summary = site
    assert summary["accepted"] and summary["gate_findings"] == 0
    assert summary["n_shard_windows"] == info["n_shard_expected"] == 12 and summary["n_shards"] == 3
    files = SC.list_release_files(roots.release)
    assert all(SC.is_allowed(f) for f in files), [f for f in files if not SC.is_allowed(f)]
    assert (roots.release / ".gate_ok").exists() and (roots.release / "LICENSE").read_text().startswith("Vehicle_Power_Type_Trajectory_Dataset")
    assert len(list((roots.release / "schema").glob("*.json"))) == 7


def test_data_is_hardlinked_and_indexed(site):
    root, roots, *_ = site
    drives = pd.read_parquet(roots.release / "index" / "drives.parquet")
    assert drives["file"].str.startswith("data/vehicle_").all()
    for f in drives["file"]:
        src = roots.build / f
        dst = roots.release / f
        assert dst.exists() and os.stat(src).st_ino == os.stat(dst).st_ino
    assert pq.read_schema(roots.release / "index" / "windows_60s.parquet").names == S.names(S.WINDOWS_INDEX_COLUMNS)
    veh = pd.read_parquet(roots.release / "index" / "vehicles.parquet")
    plat = pd.read_parquet(roots.release / "index" / "platforms.parquet")
    assert len(veh) == 4 and len(plat) == 4 and set(plat["split_platform"]) == {"train", "validation", "test"}
    assert int(veh["n_drives"].sum()) == len(drives) and int(plat["n_windows_eligible"].sum()) == 16


def test_shard_contents_match_drive_tables(site):
    root, roots, *_ = site
    windows = pd.read_parquet(roots.release / "index" / "windows_60s.parquet").set_index("window_id")
    seen = set()
    for split in ("train", "validation", "test"):
        for p in sorted((roots.release / "windows" / split).glob("*.parquet")):
            t = pq.read_table(p)
            assert t.schema.field("v_mps").type.list_size == C.WINDOW_SAMPLES
            df = t.to_pandas()
            assert (df["split_platform"] == split).all()
            for _, row in df.iterrows():
                seen.add(row["window_id"])
                drive = pq.read_table(roots.release / "data" / row["vehicle_id"] / f"{row['drive_id']}.parquet",
                                      columns=["v_mps", "gas_pressed"]).to_pandas()
                ref = drive["v_mps"].to_numpy(dtype=np.float32)[int(row["row0"]):int(row["row1"])]
                assert np.allclose(np.asarray(row["v_mps"], dtype=np.float32), ref)
                gp = np.asarray(row["gas_pressed"], dtype=np.float32)
                assert set(np.unique(gp)) <= {0.0, 1.0}
                assert np.array_equal(gp.astype(bool), drive["gas_pressed"].to_numpy()[int(row["row0"]):int(row["row1"])])
    el = windows[windows["eligible"]]
    expected = set(el[(el["task_size_eligible"] | el["task_bev_ice_eligible"])].index)
    assert seen == expected and not any(w.startswith("drive_00000" + "7") or w.startswith("drive_00000" + "8") for w in seen)


def test_checksums_cover_everything_and_rerun_is_idempotent(site):
    root, roots, *_ = site
    listed = SC.parse_checksums((roots.release / "CHECKSUMS.sha256").read_text())
    files = set(SC.checksum_members(SC.list_release_files(roots.release)))
    assert set(listed) == files and "CHECKSUMS.sha256" not in listed and ".gate_ok" not in listed
    before = (roots.release / "CHECKSUMS.sha256").read_text()
    summary = package(roots, per_shard=4, workers=1, run_gate=False)
    assert summary["shards"]["kept"] == 3 and summary["shards"]["written"] == 0
    assert summary["data"]["kept"] == summary["n_drives"] and summary["data"]["linked"] == 0
    assert summary["checksums"]["cached"] > 0
    assert (roots.release / "CHECKSUMS.sha256").read_text() == before


def test_split_col_switch_and_dry_run(site):
    root, roots, *_ = site
    alt = Roots(root, roots.build, root / "release_device", roots.private)
    summary = package(alt, split_col="split_device", per_shard=4, workers=1, run_gate=False)
    windows = pd.read_parquet(alt.release / "index" / "windows_60s.parquet").set_index("window_id")
    for split in ("train", "validation", "test"):
        for p in (alt.release / "windows" / split).glob("*.parquet"):
            ids = pq.read_table(p, columns=["window_id"]).to_pandas()["window_id"]
            assert (windows.loc[ids, "split_device"] == split).all()
    assert json.loads((alt.release / "index" / "readme_numbers.json").read_text())["split_col"] == "split_device"
    dry = Roots(root, roots.build, root / "release_dry", roots.private)
    s = package(dry, per_shard=4, workers=1, dry_run=True)
    assert s["accepted"] and s["n_shards"] == 3 and not dry.release.exists()


def test_acceptance_without_splits(site):
    root, roots, *_ = site
    build2 = root / "build_nosplit"
    shutil.copytree(roots.build, build2)
    w = pd.read_parquet(build2 / "index" / "windows.parquet")
    w["split_platform"] = None
    w["split_device"] = None
    S.write_public_parquet(w, build2 / "index" / "windows.parquet", S.WINDOWS_INDEX_COLUMNS, "windows")
    r2 = Roots(root, build2, root / "release_nosplit", roots.private)
    with pytest.raises(AcceptanceError):
        package(r2, per_shard=4, workers=1, run_gate=False)
    s = package(r2, per_shard=4, workers=1, run_gate=False, allow_no_splits=True)
    assert s["accepted"] and s["n_shards"] == 0 and not (r2.release / "windows").exists()


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(REPO / "scripts" / name), *args], capture_output=True, text=True, env=env)


def test_cli_scripts(site):
    root, roots, *_ = site
    r = run_script("s09_package.py", "--data-root", str(root), "--shard-windows", "4", "--workers", "1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "accepted=True" in r.stdout
    r = run_script("s09b_gate.py", "--data-root", str(root), "--workers", "1")
    assert r.returncode == 0 and "findings=0" in r.stdout, r.stdout + r.stderr
    (roots.release / "notes.txt").write_text("x\n")
    write_checksums(roots, workers=1, use_cache=False)
    r = run_script("s09b_gate.py", "--data-root", str(root), "--workers", "1")
    assert r.returncode == 2 and "G0/ALLOWLIST: 1" in r.stdout, r.stdout + r.stderr
    (roots.release / "notes.txt").unlink()
    write_checksums(roots, workers=1, use_cache=False)
    r = run_script("s09b_gate.py", "--data-root", str(root), "--workers", "1")      # restores .gate_ok (shared fixture)
    assert r.returncode == 0 and (roots.release / ".gate_ok").exists(), r.stdout + r.stderr
    r = run_script("s09_package.py", "--data-root", str(root), "--dry-run")
    assert r.returncode == 0 and "shards=3" in r.stdout


def _relabel_build(build: Path, old: str, new: str) -> int:
    """Rewrite the two index tables with powertrain `old` -> `new`; returns the number of changed window rows."""
    d = pd.read_parquet(build / "index" / "drives.parquet")
    w = pd.read_parquet(build / "index" / "windows.parquet")
    md, mw = d["powertrain"] == old, w["powertrain"] == old
    d.loc[md, "powertrain"] = new
    d.loc[md, "task_bev_ice_eligible"] = False
    w.loc[mw, "powertrain"] = new
    w.loc[mw, "task_bev_ice_eligible"] = False
    S.write_public_parquet(d, build / "index" / "drives.parquet", S.DRIVES_INDEX_COLUMNS, "drives")
    S.write_public_parquet(w, build / "index" / "windows.parquet", S.WINDOWS_INDEX_COLUMNS, "windows")
    return int(mw.sum())


def _shard_labels(release: Path) -> pd.DataFrame:
    parts = [pq.read_table(p, columns=["window_id", "powertrain"]).to_pandas() for p in sorted((release / "windows").rglob("*.parquet"))]
    return pd.concat(parts, ignore_index=True).set_index("window_id")


def test_relabel_without_force_rebuilds_only_the_affected_shards(site):
    root, roots, *_ = site
    build2 = root / "build_relabel"
    shutil.copytree(roots.build, build2)
    r2 = Roots(root, build2, root / "release_relabel", roots.private)
    first = package(r2, per_shard=4, workers=1, run_gate=False)
    assert first["shards"]["written"] == 3 and (r2.release / ".cache" / "vtt_shards.json").exists()
    again = package(r2, per_shard=4, workers=1, run_gate=False)
    assert again["shards"] == {**again["shards"], "kept": 3, "written": 0}
    n_changed = _relabel_build(build2, "BEV", "PHEV")                        # the train platform: size task keeps its windows
    assert n_changed > 0
    third = package(r2, per_shard=4, workers=1, run_gate=False)
    assert third["shards"]["written"] == 1 and third["shards"]["kept"] == 2, third["shards"]
    windows = pd.read_parquet(r2.release / "index" / "windows_60s.parquet").set_index("window_id")
    labels = _shard_labels(r2.release)
    assert len(labels) == 12 and (labels["powertrain"] == windows.loc[labels.index, "powertrain"]).all()
    assert (labels["powertrain"] == "PHEV").sum() == 4 and not (labels["powertrain"] == "BEV").any()
    # a rewritten drive table (same rows, new mtime) also invalidates the shards that read it
    drives = pd.read_parquet(build2 / "index" / "drives.parquet")
    f = build2 / drives["file"].iloc[0]
    t = pq.read_table(f)
    pq.write_table(t, f, compression="zstd")
    fourth = package(r2, per_shard=4, workers=1, run_gate=False)
    assert fourth["shards"]["written"] == 1 and fourth["shards"]["kept"] == 2, fourth["shards"]
    # --force rebuilds everything
    assert package(r2, per_shard=4, workers=1, run_gate=False, force=True)["shards"]["written"] == 3


def test_same_file_compares_nanosecond_mtimes(site):
    from vtt.package import _same_file
    root, *_ = site
    d = root / "same_file"
    d.mkdir(exist_ok=True)
    a, b = d / "a.bin", d / "b.bin"
    a.write_bytes(b"x" * 10)
    b.write_bytes(b"y" * 10)
    st = a.stat()
    os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert _same_file(a, b)                                                    # an exact copy keeps size and ns mtime
    os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    assert not _same_file(a, b)                                                # same integer second, different file
    c = d / "c.bin"
    os.link(a, c)
    assert _same_file(a, c) and not _same_file(a, d / "missing.bin")


def test_cli_errors_exit_2_without_traceback(site):
    root, roots, *_ = site
    build3 = root / "build_broken"
    shutil.copytree(roots.build, build3)
    next((build3 / "data").rglob("*.parquet")).unlink()                        # a drive table listed in the index is missing
    args = ("--data-root", str(root), "--build-root", str(build3), "--release-root", str(root / "release_broken"),
            "--shard-windows", "4", "--workers", "1", "--skip-gate")
    r = run_script("s09_package.py", *args)
    assert r.returncode == 2 and "Traceback" not in r.stderr and "s09: FAILED (FileNotFoundError)" in r.stdout, r.stdout + r.stderr
    r = run_script("s09_package.py", *args, "--debug")
    assert r.returncode == 1 and "Traceback" in r.stderr
