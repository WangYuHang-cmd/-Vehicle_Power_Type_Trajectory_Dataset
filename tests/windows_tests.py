"""S6 window index: counts, eligibility reasons, statistics, human_only_strict guard, label copy, script round trip."""
from __future__ import annotations

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

from vtt import config as C
from vtt import schema as S
from vtt import windows as W
from tests import synth_drive as SD

R = Path(__file__).resolve().parents[1]
TMP_BASE = Path(C.DATA_ROOT) / "tmp"
DT = C.NOMINAL_DT_S
N = C.WINDOW_SAMPLES


@pytest.fixture(scope="module")
def tree():
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_windows_", dir=TMP_BASE))
    rows = []
    # drive A: 3 full windows + a 50-row tail; window 1 has a gap row; window 2 is stationary; DAS in window 0 only
    n = 3 * N + 50
    v = np.full(n, 15.0)
    v[2 * N:3 * N] = 0.0
    das = np.zeros(n, bool)
    das[100:200] = True
    lead = np.zeros(n, bool)
    lead[0:N] = True
    slope = np.where(lead, 0.0, np.nan)
    slope[300:320] = 1.0                      # one lead acceleration onset at row 300 of window 0
    dfa = SD.make_drive(n, v=v, das_any=das, gap_rows=[N + 10], lead_valid=lead, lead_a_slope=slope)
    SD.write_drive(dfa, root, "vehicle_0001", "drive_000001")
    rows.append(SD.drives_index_row(dfa, "vehicle_0001", "drive_000001", "group_0001", platform="PLAT_A", powertrain="BEV",
                                    powertrain_raw="BEV", size_class="SMALL"))
    # drive B: 2 full windows, DAS active in the last 5 s of window 0 -> window 1 human_only but not strict
    n = 2 * N
    das = np.zeros(n, bool)
    das[N - 50:N] = True
    dfb = SD.make_drive(n, v=np.full(n, 20.0), das_any=das)
    SD.write_drive(dfb, root, "vehicle_0002", "drive_000002")
    rows.append(SD.drives_index_row(dfb, "vehicle_0002", "drive_000002", "group_0002", platform="PLAT_B", powertrain="ICE",
                                    powertrain_raw="ICE", size_class="LARGE"))
    # drive C: short drive (200 rows) -> one 'short' window; DAS the whole time -> kind das
    dfc = SD.make_drive(200, v=np.full(200, 10.0), das_any=np.ones(200, bool))
    SD.write_drive(dfc, root, "vehicle_0002", "drive_000003")
    rows.append(SD.drives_index_row(dfc, "vehicle_0002", "drive_000003", "group_0002", platform="PLAT_B", powertrain="ICE",
                                    powertrain_raw="ICE", size_class="LARGE"))
    # drive D: acceleration ramps -> episode counts
    v = SD.ramp_speed(N, [(10, 0.0), (6, 6.0), (20, 6.0), (5, 0.0), (19, 0.0)])
    dfd = SD.make_drive(N, v=v)
    SD.write_drive(dfd, root, "vehicle_0003", "drive_000004")
    rows.append(SD.drives_index_row(dfd, "vehicle_0003", "drive_000004", "group_0003", platform="PLAT_C", powertrain=None,
                                    powertrain_raw="MIXED", size_class=None))
    SD.write_drives_index(rows, root)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _two_drive_tree(prefix: str, rows_per_drive: int = N) -> Path:
    """vehicle_0001 with drive_000001 and drive_000002 (one full moving window each)."""
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=TMP_BASE))
    rows = []
    for k in (1, 2):
        df = SD.make_drive(rows_per_drive, v=np.full(rows_per_drive, 15.0 + k))
        SD.write_drive(df, root, "vehicle_0001", f"drive_00000{k}")
        rows.append(SD.drives_index_row(df, "vehicle_0001", f"drive_00000{k}", "group_0001"))
    SD.write_drives_index(rows, root)
    return root


def test_window_bounds():
    assert W.window_bounds(0) == []
    assert W.window_bounds(600) == [(0, 600)]
    assert W.window_bounds(650) == [(0, 600), (600, 650)]
    assert W.window_bounds(1800) == [(0, 600), (600, 1200), (1200, 1800)]


def test_build_windows_counts_and_eligibility(tree):
    roots = W.Roots(tree)
    drives = pd.read_parquet(roots.drives)
    windows, counts = W.build_windows(drives, roots, workers=1)
    assert counts["windows"] == 4 + 2 + 1 + 1 and counts["drives_with_windows"] == 4
    a = windows[windows.drive_id == "drive_000001"].set_index("w_idx")
    assert a.loc[0, "eligible"] and a.loc[0, "inelig_reason"] is None
    assert not a.loc[1, "eligible"] and a.loc[1, "inelig_reason"] == "gap" and a.loc[1, "gap_count"] == 1
    assert not a.loc[2, "eligible"] and a.loc[2, "inelig_reason"] == "stationary"
    assert not a.loc[3, "eligible"] and a.loc[3, "inelig_reason"] == "short" and a.loc[3, "row1"] - a.loc[3, "row0"] == 50
    assert a.loc[0, "kind"] == "mixed" and abs(a.loc[0, "das_any_frac"] - 100 / 600) < 1e-6
    assert "span" not in W.INELIG_REASONS and counts["stale_rows_dropped"] == 0
    assert a.loc[1, "kind"] == "human" and a.loc[1, "human_only"]
    assert a.loc[0, "n_lead_accel_events"] == 1 and abs(a.loc[0, "lead_frac"] - 1.0) < 1e-9
    assert abs(a.loc[0, "thw_med"] - 2.0) < 1e-6 and abs(a.loc[0, "cf_frac"] - 1.0) < 1e-9
    assert a.loc[1, "n_lead_accel_events"] == 0 and np.isnan(a.loc[1, "thw_med"])
    assert abs(a.loc[0, "span_s"] - 59.9) < 1e-6 and abs(a.loc[1, "t0_s"] - 60.0) < 1e-6
    c = windows[windows.drive_id == "drive_000003"].iloc[0]
    assert c["kind"] == "das" and not c["eligible"] and c["inelig_reason"] == "short" and not c["human_only"]
    assert counts["eligible"] == int(windows["eligible"].sum()) == 4      # A0, B0, B1, D0


def test_human_only_strict_guard(tree):
    roots = W.Roots(tree)
    drives = pd.read_parquet(roots.drives)
    windows, _ = W.build_windows(drives, roots, workers=1)
    b = windows[windows.drive_id == "drive_000002"].set_index("w_idx")
    assert not b.loc[0, "human_only"] and not b.loc[0, "human_only_strict"]
    assert b.loc[1, "human_only"] and not b.loc[1, "human_only_strict"]       # DAS 5 s before the window
    a = windows[windows.drive_id == "drive_000001"].set_index("w_idx")
    assert a.loc[1, "human_only_strict"]                                       # DAS ended 40 s before window 1
    d = windows[windows.drive_id == "drive_000004"].iloc[0]
    assert d["human_only_strict"]                                              # first window: vacuous guard
    human = W.human_only_subset(windows)
    assert set(human.window_id) == {"drive_000004_w0"}


def test_episode_counts_and_label_copy(tree):
    roots = W.Roots(tree)
    drives = pd.read_parquet(roots.drives)
    windows, _ = W.build_windows(drives, roots, workers=1)
    d = windows[windows.drive_id == "drive_000004"].iloc[0]
    assert d["n_accel_ep"] == 1 and d["n_decel_ep"] == 1
    assert d["v_max"] > 5.9 and d["a_max"] > 0.9 and d["a_min"] < -1.0
    assert d["platform"] == "PLAT_C" and d["powertrain"] is None and d["powertrain_raw"] == "MIXED"
    assert not d["task_bev_ice_eligible"] and not d["task_size_eligible"]
    a = windows[windows.drive_id == "drive_000001"].iloc[0]
    assert a["group_id"] == "group_0001" and a["powertrain"] == "BEV" and a["size_class"] == "SMALL" and a["task_bev_ice_eligible"]
    assert a["split_device"] is None and bool(a["low_support_cell"]) is False     # bool columns cannot be null in the public schema
    assert list(windows.columns) == S.names(S.WINDOWS_INDEX_COLUMNS)
    assert W.acceptance_problems(windows, drives) == []


def test_script_round_trip_and_resume(tree):
    env = {"PYTHONPATH": str(R), "VTT_DATA_ROOT": str(tree), "PATH": "/usr/bin:/bin"}
    cmd = [sys.executable, str(R / "scripts" / "s06_windows.py"), "--data-root", str(tree), "--workers", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "windows=8" in r.stdout and "eligible=4" in r.stdout
    roots = W.Roots(tree)
    assert S.check_public_parquet(roots.windows, S.WINDOWS_INDEX_COLUMNS, "windows") == []
    assert S.check_public_parquet(roots.human_only_windows, S.WINDOWS_INDEX_COLUMNS, "windows") == []
    w = pd.read_parquet(roots.windows)
    h = pd.read_parquet(roots.human_only_windows)
    assert len(w) == 8 and len(h) == 1 and set(h.window_id) <= set(w.window_id)
    assert h["human_only_strict"].all() and h["eligible"].all()
    meta = pq.read_schema(roots.windows).metadata
    assert set(meta) == {b"vtt_table", b"vtt_schema_version"}
    # resume: parts are cached per vehicle and reused on the second run
    parts = sorted(roots.windows_parts.glob("*.parquet"))
    assert len(parts) == 3
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r2.returncode == 0 and "parts_reused=3" in r2.stdout and "parts_to_build=0" in r2.stdout
    # dry run writes nothing new and prints counts
    mt = roots.windows.stat().st_mtime
    r3 = subprocess.run(cmd + ["--dry-run", "--force"], capture_output=True, text=True, env=env)
    assert r3.returncode == 0 and "dry-run" in r3.stdout and roots.windows.stat().st_mtime == mt
    # no identifier-like token in the output
    assert "<hex16>" not in r.stdout
    from vtt.private import mask
    assert mask(r.stdout) == r.stdout


def test_acceptance_detects_bad_rows(tree):
    roots = W.Roots(tree)
    drives = pd.read_parquet(roots.drives)
    windows, _ = W.build_windows(drives, roots, workers=1)
    bad = windows.copy()
    bad.loc[bad.index[0], "gap_count"] = 3
    bad.loc[bad.index[0], "eligible"] = True
    assert any("gap" in p for p in W.acceptance_problems(bad, drives))
    dup = pd.concat([windows, windows.head(1)], ignore_index=True)
    assert any("duplicate" in p for p in W.acceptance_problems(dup, drives))
    # a ghost window (drive no longer in the index, hence null group_id) and a vehicle mismatch are acceptance failures
    ghost = windows.copy()
    ghost.loc[ghost.index[0], ["drive_id", "window_id", "group_id"]] = ["drive_009999", "drive_009999_w0", None]
    probs = W.acceptance_problems(ghost, drives)
    assert any("not in the index" in p and "--force" in p for p in probs)
    moved = windows.copy()
    moved.loc[moved.index[0], "vehicle_id"] = "vehicle_0099"
    assert any("vehicle_id differs" in p for p in W.acceptance_problems(moved, drives))
    nogroup = windows.copy()
    nogroup.loc[nogroup.index[0], "group_id"] = None
    assert any("group_id" in p for p in W.acceptance_problems(nogroup, drives))


def test_stale_parts_never_leak_removed_drives():
    """A drive removed from the index (file deleted) must not survive in the cached per-vehicle part."""
    root = _two_drive_tree("vtt_windows_stale_")
    try:
        roots = W.Roots(root)
        drives = pd.read_parquet(roots.drives)
        logs: list[str] = []
        w1, c1 = W.build_windows(drives, roots, workers=1, log=logs.append)
        assert c1["windows"] == 2 and W.acceptance_problems(w1, drives) == []
        os.remove(roots.data_dir / "vehicle_0001" / "drive_000002.parquet")
        SD.write_drives_index([SD.drives_index_row(SD.make_drive(N, v=np.full(N, 16.0)), "vehicle_0001", "drive_000001", "group_0001")], root)
        drives2 = pd.read_parquet(roots.drives)
        logs.clear()
        w2, c2 = W.build_windows(drives2, roots, workers=1, log=logs.append)
        assert "parts_reused=1" in logs[0]                       # the superset part is still valid for drive 1 ...
        assert set(w2.drive_id) == {"drive_000001"} and c2["windows"] == 1 and c2["stale_rows_dropped"] == 1
        assert w2["group_id"].notna().all() and W.acceptance_problems(w2, drives2) == []   # ... but drive 2 is gone
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_limit_smoke_part_is_rebuilt_by_the_full_run():
    """A part written from a --limit subset covers only some drives of the vehicle: the next full run rebuilds it."""
    root = _two_drive_tree("vtt_windows_limit_")
    try:
        roots = W.Roots(root)
        drives = pd.read_parquet(roots.drives)
        logs: list[str] = []
        w1, _ = W.build_windows(drives.head(1), roots, workers=1, log=logs.append)
        assert set(w1.drive_id) == {"drive_000001"}
        logs.clear()
        w2, c2 = W.build_windows(drives, roots, workers=1, log=logs.append)
        assert "parts_to_build=1" in logs[0] and "parts_reused=0" in logs[0]
        assert set(w2.drive_id) == {"drive_000001", "drive_000002"} and W.acceptance_problems(w2, drives) == []
        logs.clear()
        w3, _ = W.build_windows(drives.head(1), roots, workers=1, log=logs.append)   # a later smoke run reuses the full part
        assert "parts_reused=1" in logs[0] and set(w3.drive_id) == {"drive_000001"} and W.acceptance_problems(w3, drives.head(1)) == []
        # the script's acceptance message for a poisoned cache names the remedy
        part = roots.windows_parts / "vehicle_0001.parquet"
        poisoned = pd.read_parquet(part)
        poisoned = poisoned[poisoned.drive_id == "drive_000001"]
        poisoned.to_parquet(part, index=False)
        os.utime(part, None)
        assert W.part_is_fresh(part, [roots.data_dir / "vehicle_0001" / f"drive_00000{k}.parquet" for k in (1, 2)]) is True
        assert W.part_is_fresh(part, [roots.data_dir / "vehicle_0001" / f"drive_00000{k}.parquet" for k in (1, 2)],
                               ["drive_000001", "drive_000002"]) is False
        assert W.part_drive_ids(part) == {"drive_000001"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_drive_moved_to_another_vehicle_appears_once():
    """A drive re-keyed to another vehicle (S4 re-key) keeps its drive_id; the old vehicle's part still lists it but
    the assembled index carries it once, under the new vehicle."""
    root = _two_drive_tree("vtt_windows_moved_")
    try:
        roots = W.Roots(root)
        drives = pd.read_parquet(roots.drives)
        W.build_windows(drives, roots, workers=1, log=lambda *a: None)
        src = roots.data_dir / "vehicle_0001" / "drive_000002.parquet"
        dst = roots.data_dir / "vehicle_0002" / "drive_000002.parquet"
        dst.parent.mkdir()
        shutil.move(src, dst)
        drives.loc[drives.drive_id == "drive_000002", ["vehicle_id", "file"]] = ["vehicle_0002", "data/vehicle_0002/drive_000002.parquet"]
        S.write_public_parquet(drives, roots.drives, S.DRIVES_INDEX_COLUMNS, "drives")
        drives2 = pd.read_parquet(roots.drives)
        w, c = W.build_windows(drives2, roots, workers=1, log=lambda *a: None)
        assert c["windows"] == 2 and c["stale_rows_dropped"] == 1 and not w["window_id"].duplicated().any()
        assert w.set_index("drive_id").loc["drive_000002", "vehicle_id"] == "vehicle_0002"
        assert W.acceptance_problems(w, drives2) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
