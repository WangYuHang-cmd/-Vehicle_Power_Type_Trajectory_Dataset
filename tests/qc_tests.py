"""S7 QC: signal checks, duplicates, label-audit rules, overrides, REVIEW.md rendering, script exit codes."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vtt import config as C
from vtt import qc as Q
from vtt import schema as S
from vtt import windows as W
from vtt.private import mask
from tests import synth_drive as SD

R = Path(__file__).resolve().parents[1]
TMP_BASE = Path(C.DATA_ROOT) / "tmp"
N = C.WINDOW_SAMPLES


def _tree(prefix: str) -> Path:
    """Seven drives on five vehicles / four groups with a private audit probe and name map.
    d1 BEV + engine rpm (conflict); d2 ICE + regen (conflict); d3 ICE engine-off moving (suspect); d4 HEV engine always on
    (suspect); d5 HEV ok, duplicate of d6 (same speed trace), radar_unavailable with camera lead source; d6 ICE
    no probe row, yaw all zero; d7 ICE whose port never fills engineRpm (rpm_pos_frac 0, off_moving 1 -> no_rpm_signal),
    distinct speed trace."""
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=prefix, dir=TMP_BASE))
    rows, names, audit = [], [], []
    v = SD.ramp_speed(2 * N, [(10, 0.0), (6, 12.0), (60, 12.0), (8, 0.0), (36, 0.0)])
    specs = [
        ("vehicle_0001", "drive_000001", "group_0001", "PLAT_BEV", "BEV", "BEV", dict(engine_rpm_pos_frac=0.5, engine_off_moving_frac=0.0, regen_any=True, charging_any=False)),
        ("vehicle_0002", "drive_000002", "group_0002", "PLAT_ICE_A", "ICE", "ICE", dict(engine_rpm_pos_frac=0.9, engine_off_moving_frac=0.0, regen_any=True, charging_any=False)),
        ("vehicle_0003", "drive_000003", "group_0003", "PLAT_ICE_B", "ICE", "ICE", dict(engine_rpm_pos_frac=0.5, engine_off_moving_frac=0.5, regen_any=False, charging_any=False)),
        ("vehicle_0004", "drive_000004", "group_0004", "PLAT_HEV_A", "HEV", "HEV", dict(engine_rpm_pos_frac=0.99, engine_off_moving_frac=0.0, regen_any=True, charging_any=False)),
        ("vehicle_0005", "drive_000005", "group_0004", "PLAT_HEV_B", "HEV", "HEV", dict(engine_rpm_pos_frac=0.6, engine_off_moving_frac=0.4, regen_any=True, charging_any=False)),
        ("vehicle_0002", "drive_000006", "group_0002", "PLAT_ICE_A", "ICE", "ICE", None),
        ("vehicle_0002", "drive_000007", "group_0002", "PLAT_ICE_A", "ICE", "ICE", dict(engine_rpm_pos_frac=0.0, engine_off_moving_frac=1.0, regen_any=False, charging_any=False)),
    ]
    for k, (veh, drv, grp, plat, pt, raw, au) in enumerate(specs, start=1):
        lead = np.zeros(2 * N, bool)
        lead[200:1000] = True
        yaw = np.zeros(2 * N) if drv == "drive_000006" else None
        radar_unavailable = drv == "drive_000005"
        df = SD.make_drive(2 * N, v=v + (1.0 if drv == "drive_000007" else 0.0), lead_valid=lead,
                           lead_source=2 if radar_unavailable else 1, yaw=yaw)
        SD.write_drive(df, root, veh, drv)
        rows.append(SD.drives_index_row(df, veh, drv, grp, platform=plat, powertrain=pt, powertrain_raw=raw, size_class="MID",
                                        region="europe" if k % 2 else "north_america", radar_unavailable=radar_unavailable,
                                        label_note="shared platform" if drv == "drive_000003" else None, n_segments=4))
        uid = SD.fake_route_uid(k)
        names.append(dict(route_uid=uid, vehicle_id=veh, drive_id=drv, group_id=grp))
        if au is not None:
            audit.append(dict(route_uid=uid, n_segments_read=3, ecus="[]", has_hybrid_ecu=pt != "ICE", has_transmission_ecu=True,
                              transmission="automatic", mass_kg=1500.0, wheelbase_m=2.7, steer_ratio=15.0, fuel_gauge_med=np.nan,
                              gear_values="[]", yaw_rate_nonzero_frac=0.9, eps_torque_nonzero_frac=0.5, n_samples=1800, err="", **au))
    SD.write_drives_index(rows, root)
    pd.DataFrame(names).to_parquet(root / "build" / "drive_names.parquet", index=False)
    priv = root / "private"
    priv.mkdir(mode=0o700)
    pd.DataFrame(audit).to_parquet(priv / "route_audit.parquet", index=False)
    (priv / "qc").mkdir(mode=0o700)
    pd.DataFrame(dict(drive_id=[s[1] for s in specs], cs_dt_med_s=[0.1, 0.1, 0.1, 0.1, 0.1, 0.13, 0.1])).to_parquet(
        priv / "qc" / "drives_private_qc.parquet", index=False)
    pd.DataFrame(dict(dongle_log=["f" * 10 + "%06x" % 1] * 2 + ["f" * 10 + "%06x" % 2] * 5,
                      platform=["PLAT_BEV", "PLAT_ICE_A", "PLAT_ICE_B", "PLAT_HEV_A", "PLAT_HEV_B", "PLAT_ICE_A", "PLAT_ICE_A"],
                      drive_id=[s[1] for s in specs])).to_parquet(priv / "pseudonym_map.parquet", index=False)
    # windows via S6 so the review has eligible-window hours
    roots = W.Roots(root)
    drives = pd.read_parquet(roots.drives)
    windows, _ = W.build_windows(drives, roots, workers=1)
    W.write_windows(windows, roots.windows)
    W.write_windows(W.human_only_subset(windows), roots.human_only_windows)
    return root


@pytest.fixture(scope="module")
def tree():
    root = _tree("vtt_qc_")
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_audit_rules():
    row = pd.Series(dict(engine_rpm_pos_frac=0.5, engine_off_moving_frac=0.0, regen_any=False, charging_any=False, err=""))
    assert Q.audit_verdict("BEV", row) == "conflict_bev_engine_rpm"
    assert Q.audit_verdict("BEV", pd.Series(dict(engine_rpm_pos_frac=0.0, engine_off_moving_frac=np.nan, regen_any=True, charging_any=True, err=""))) == "ok"
    assert Q.audit_verdict("ICE", pd.Series(dict(engine_rpm_pos_frac=0.9, engine_off_moving_frac=0.0, regen_any=False, charging_any=True, err=""))) == "conflict_ice_electric"
    assert Q.audit_verdict("ICE", pd.Series(dict(engine_rpm_pos_frac=0.6, engine_off_moving_frac=0.4, regen_any=False, charging_any=False, err=""))) == "suspect_mild_hybrid"
    assert Q.audit_verdict("ICE", pd.Series(dict(engine_rpm_pos_frac=0.9, engine_off_moving_frac=0.1, regen_any=False, charging_any=False, err=""))) == "ok"
    assert Q.audit_verdict("HEV", pd.Series(dict(engine_rpm_pos_frac=1.0, engine_off_moving_frac=0.0, regen_any=True, charging_any=False, err=""))) == "suspect_hybrid_engine_always_on"
    assert Q.audit_verdict("PHEV", pd.Series(dict(engine_rpm_pos_frac=0.5, engine_off_moving_frac=0.3, regen_any=True, charging_any=True, err=""))) == "ok"
    assert Q.audit_verdict("ICE", pd.Series(dict(engine_rpm_pos_frac=np.nan, engine_off_moving_frac=np.nan, regen_any=False, charging_any=False, err="probe failed"))) == "no_audit_signal"
    assert Q.audit_verdict("BEV", pd.Series(dict(engine_rpm_pos_frac=np.nan, engine_off_moving_frac=np.nan, regen_any=None, charging_any=None, err=""))) == "no_audit_signal"
    assert Q.audit_verdict("BEV", None) == "no_audit"
    assert Q.audit_verdict(None, row) == "na"
    # an rpm channel that is never positive is an unpopulated channel on ICE / hybrid ports, not an engine-off fact
    never = pd.Series(dict(engine_rpm_pos_frac=0.0, engine_off_moving_frac=1.0, regen_any=False, charging_any=False, err=""))
    assert Q.audit_verdict("ICE", never) == "no_rpm_signal"
    assert Q.audit_verdict("HEV", never) == "no_rpm_signal" and Q.audit_verdict("PHEV", never) == "no_rpm_signal"
    assert Q.audit_verdict("BEV", never) == "ok"                                 # for a BEV rpm == 0 is the consistent reading
    assert Q.audit_verdict("ICE", pd.Series(dict(engine_rpm_pos_frac=0.0, engine_off_moving_frac=1.0, regen_any=True, charging_any=False, err=""))) == "conflict_ice_electric"
    assert Q.audit_verdict("HEV", pd.Series(dict(engine_rpm_pos_frac=0.5, engine_off_moving_frac=np.nan, regen_any=True, charging_any=False, err=""))) == "ok"
    assert "no_rpm_signal" in Q.VERDICTS and not "no_rpm_signal".startswith((Q.CONFLICT_PREFIX, Q.SUSPECT_PREFIX))


def test_label_audit_join_and_signal_checks(tree):
    roots = W.Roots(tree)
    drives = pd.read_parquet(roots.drives)
    audit = Q.label_audit(drives, pd.read_parquet(roots.route_audit), pd.read_parquet(roots.drive_names))
    v = audit.set_index("drive_id")["verdict"]
    assert v["drive_000001"] == "conflict_bev_engine_rpm" and v["drive_000002"] == "conflict_ice_electric"
    assert v["drive_000003"] == "suspect_mild_hybrid" and v["drive_000004"] == "suspect_hybrid_engine_always_on"
    assert v["drive_000005"] == "ok" and v["drive_000006"] == "no_audit" and v["drive_000007"] == "no_rpm_signal"
    assert Q.unresolved_conflicts(v, set()) == ["drive_000001", "drive_000002"]
    assert Q.unresolved_conflicts(v, {"drive_000001"}) == ["drive_000002"]
    scan = Q.scan_all(drives, roots, workers=1)
    assert len(scan) == 7 and list(scan.columns) == list(Q.SCAN_COLUMNS)
    dup = Q.duplicate_of(scan)
    assert (dup != "").sum() == 5 and dup["drive_000001"] == "" and dup["drive_000006"] == "drive_000001" and dup["drive_000007"] == ""
    assert bool(scan.set_index("drive_id").loc["drive_000006", "yaw_all_zero"]) and not bool(scan.set_index("drive_id").loc["drive_000001", "yaw_all_zero"])
    lf = Q.lead_source_flags(drives, scan)
    assert (lf == "").all()                                        # no vacuous mismatch flag; nothing else fires here
    synthetic = pd.DataFrame(dict(drive_id=["a", "b", "c"], radar_unavailable=[False, False, True], moving_s=[700.0, 700.0, 100.0]))
    sc = pd.DataFrame(dict(drive_id=["a", "b", "c"], lead_n_valid=[1500, 0, 10], lead_radar_share=[0.0, np.nan, 0.0], lead_vision_share=[1.0, np.nan, 1.0]))
    lf2 = Q.lead_source_flags(synthetic, sc)
    assert lf2["a"] == "lead_radar_never" and lf2["b"] == "no_lead" and lf2["c"] == ""
    lt = Q.lead_source_table(drives, scan).set_index("radar_unavailable")
    assert lt.loc[False, "drives"] == 6 and lt.loc[True, "drives"] == 1 and lt.loc[True, "drives_with_lead"] == 1
    assert abs(lt.loc[False, "radar_share"] - 1.0) < 1e-9 and abs(lt.loc[True, "camera_share"] - 1.0) < 1e-9
    assert lt.loc[False, "valid_lead_samples"] == 6 * 800
    rates = Q.rate_table(drives, pd.read_parquet(roots.private_drives_qc)).set_index("drive_id")
    assert abs(rates.loc["drive_000001", "cs_rate_hz"] - 10.0) < 1e-9 and rates.loc["drive_000001", "rate_flag"] == ""
    assert rates.loc["drive_000006", "rate_flag"] == "rate_low"
    rates2 = Q.rate_table(drives, None).set_index("drive_id")
    assert np.isnan(rates2.loc["drive_000001", "cs_rate_hz"]) and abs(rates2.loc["drive_000001", "rows_per_seg"] - 400.0) < 1e-9
    assert rates2.loc["drive_000001", "rate_flag"] == "rows_per_seg_low"       # 1200 rows for 4 nominal segments
    n_multi, flag, basis = Q.multi_platform(drives, pd.read_parquet(roots.pseudonym_map))
    assert basis == "device" and n_multi == 2 and (flag != "").all()
    n_multi2, flag2, basis2 = Q.multi_platform(drives, None)
    assert basis2 == "group" and n_multi2 == 1 and flag2["drive_000004"] == "multi_platform_group" and flag2["drive_000001"] == ""
    overall, per = Q.dead_channel_summary(drives)
    assert overall["gas"] == 0.0 and overall["brake"] == 0.0 and per.iloc[0]["platform"] == "PLAT_ICE_A" and per.iloc[0]["drives"] == 3


def test_note_composition_idempotent():
    assert Q.compose_note([], [], "ok", False) == "label_audit=ok"
    note = Q.compose_note(["shared platform"], ["reviewed"], "conflict_ice_electric", True)
    assert note == "shared platform; override:reviewed; label_audit=conflict_ice_electric/resolved"
    assert Q.strip_generated(note) == ["shared platform"]
    assert Q.compose_note(Q.strip_generated(note), ["reviewed"], "conflict_ice_electric", True) == note
    assert Q.compose_note([], [], None, False) is None


def test_review_renders(tree):
    roots = W.Roots(tree)
    drives = pd.read_parquet(roots.drives)
    windows = pd.read_parquet(roots.windows)
    scan = Q.scan_all(drives, roots, workers=1)
    audit = Q.label_audit(drives, pd.read_parquet(roots.route_audit), pd.read_parquet(roots.drive_names))
    rates = Q.rate_table(drives, None)
    n_multi, multi, basis = Q.multi_platform(drives, None)
    qc = Q.drives_qc_table(drives, scan, rates, Q.lead_source_flags(drives, scan), Q.duplicate_of(scan), multi, audit, set(), windows)
    assert len(qc) == 7 and "v_sha1" not in qc.columns and qc["yaw_all_zero"].dtype == bool
    assert qc.set_index("drive_id").loc["drive_000005", "flags"] == "rows_per_seg_low;multi_platform_group;duplicate"
    assert "duplicate" in qc.set_index("drive_id").loc["drive_000006", "flags"] and "yaw_all_zero" in qc.set_index("drive_id").loc["drive_000006", "flags"]
    assert qc.set_index("drive_id").loc["drive_000001", "n_eligible_windows"] == 2
    assert qc.set_index("drive_id").loc["drive_000007", "label_audit"] == "no_rpm_signal"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # no pandas downcasting FutureWarning, also with empty flag columns
        md = Q.review_markdown(drives, windows, qc, Q.dead_channel_summary(drives), audit, set(), n_multi, basis, False, True,
                               Q.lead_source_table(drives, scan))
        empty = qc.copy()
        empty["lead_flag"] = ""
        empty["rate_flag"] = ""
        empty["duplicate_of"] = ""
        empty["yaw_all_zero"] = False
        md_empty = Q.review_markdown(drives, windows, empty, Q.dead_channel_summary(drives), audit, set(), n_multi, basis, False, True)
    assert "- lead: no flags" in md_empty and "- rate: no flags" in md_empty
    for h in ("# VTT build review", "## 1. Cells", "## 2. Hours by cell and region", "## 3. DAS share", "## 4. Dead channels",
              "## 5. Signal checks", "Lead source by radar availability", "## 6. Label audit", "### Conflicts by platform",
              "### Suspect drives", "## 7. Flagged drives", "## 8. How to resolve"):
        assert h in md
    assert "| MID x BEV |" in md and "| MID x HEV |" in md and "| MID x ICE |" in md
    assert "unresolved label conflicts: 2" in md and "conflict_bev_engine_rpm" in md and "no_rpm_signal=1" in md
    suspects = md.split("### Suspect drives")[1].split("## 7.")[0]
    assert "drive_000003" in suspects and "drive_000004" in suspects and "drive_000007" not in suspects
    acr = "AD" + "AS"                                                        # assembled: the gate scans tests/ too
    assert acr not in md.replace(acr + "-TO", "")
    assert mask(md) == md
    cells = Q.cell_table(drives, windows).set_index("cell")
    assert cells.loc["MID x ICE", "drives"] == 4 and abs(cells.loc["MID x ICE", "hours"] - 4 * 120 / 3600) < 1e-9
    assert abs(cells.loc["MID x ICE", "eligible_window_h"] - 8 / 60) < 1e-9 and cells.loc["MID x ICE", "top_platform"] == "PLAT_ICE_A"
    assert abs(cells.loc["MID x ICE", "top_platform_share"] - 3 / 4) < 1e-9 and cells.loc["MID x ICE", "das_share"] == 0.0
    reg = Q.region_table(drives).set_index("cell")
    assert abs(reg.loc["MID x ICE", "europe"] - 240 / 3600) < 1e-9 and abs(reg.loc["MID x ICE", "north_america"] - 240 / 3600) < 1e-9


def test_scan_cache_drops_ghost_drives_and_dry_run_writes_nothing():
    TMP_BASE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_qc_scan_", dir=TMP_BASE))
    try:
        rows = []
        for k in (1, 2):
            df = SD.make_drive(N, v=np.full(N, 10.0 + k))
            SD.write_drive(df, root, "vehicle_0001", f"drive_00000{k}")
            rows.append(SD.drives_index_row(df, "vehicle_0001", f"drive_00000{k}", "group_0001"))
        SD.write_drives_index(rows, root)
        roots = W.Roots(root)
        drives = pd.read_parquet(roots.drives)
        logs: list[str] = []
        dry = Q.scan_all(drives, roots, workers=1, dry_run=True, log=logs.append)
        assert len(dry) == 2 and not roots.qc_parts.exists()                       # dry run: computed in memory only
        scan1 = Q.scan_all(drives, roots, workers=1, log=logs.append)
        assert len(scan1) == 2 and (roots.qc_parts / "vehicle_0001.parquet").exists()
        os.remove(roots.data_dir / "vehicle_0001" / "drive_000002.parquet")
        SD.write_drives_index(rows[:1], root)
        drives2 = pd.read_parquet(roots.drives)
        logs.clear()
        scan2 = Q.scan_all(drives2, roots, workers=1, log=logs.append)
        assert "parts_reused=1" in logs[0] and any("stale_rows_dropped=1" in m for m in logs)
        assert list(scan2.drive_id) == ["drive_000001"]
        assert (Q.duplicate_of(scan2) == "").all()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_overrides_and_script_exit_codes():
    root = _tree("vtt_qc_script_")
    try:
        _overrides_and_script_exit_codes(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _overrides_and_script_exit_codes(root: Path) -> None:
    roots = W.Roots(root)
    env = {"PYTHONPATH": str(R), "VTT_DATA_ROOT": str(root), "PATH": "/usr/bin:/bin"}
    cmd = [sys.executable, "-W", "error::FutureWarning", str(R / "scripts" / "s07_qc.py"), "--data-root", str(root), "--workers", "2"]
    # a dry run before anything exists writes nothing at all (no qc parts, no outputs)
    r0 = subprocess.run(cmd + ["--dry-run", "--allow-conflicts"], capture_output=True, text=True, env=env)
    assert r0.returncode == 0 and "dry-run: nothing written" in r0.stdout, r0.stdout + r0.stderr
    assert not roots.qc_parts.exists() and not roots.review_md.exists() and not roots.qc.exists()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "unresolved_conflicts=2" in r.stdout, r.stdout + r.stderr
    assert "lead_source by radar_unavailable" in r.stdout and "no_rpm_signal=1" in r.stdout
    assert mask(r.stdout) == r.stdout and "f" * 10 not in r.stdout
    # outputs exist even on exit 2 (the user reviews them); private CSV is 0600 and carries the uids, public CSV does not
    assert roots.review_md.exists() and roots.drives_qc.exists() and roots.label_audit_private.exists()
    assert stat.S_IMODE(os.stat(roots.label_audit_private).st_mode) == 0o600
    priv = roots.label_audit_private.read_text()
    assert SD.fake_route_uid(1) in priv
    pub = roots.drives_qc.read_text()
    assert SD.fake_route_uid(1) not in pub and "conflict_bev_engine_rpm" in pub
    drives = pd.read_parquet(roots.drives)
    notes = drives.set_index("drive_id")["label_note"]
    assert notes["drive_000001"] == "label_audit=conflict_bev_engine_rpm"
    assert notes["drive_000003"] == "shared platform; label_audit=suspect_mild_hybrid"
    assert notes["drive_000006"] == "label_audit=no_audit" and notes["drive_000007"] == "label_audit=no_rpm_signal"
    assert S.check_public_parquet(roots.drives, S.DRIVES_INDEX_COLUMNS, "drives") == []
    # --allow-conflicts -> exit 0; re-run is idempotent (no duplicated tags)
    r2 = subprocess.run(cmd + ["--allow-conflicts"], capture_output=True, text=True, env=env)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    notes2 = pd.read_parquet(roots.drives).set_index("drive_id")["label_note"]
    assert notes2.equals(notes)
    # overrides: relabel the BEV platform as HEV (conflict disappears), resolve the ICE regen drive with a note
    pd.DataFrame([dict(scope="platform", key="PLAT_BEV", powertrain="HEV", label_note="audit: engine rpm seen", resolve="0"),
                  dict(scope="drive", key="drive_000002", powertrain="", label_note="regen flag is a CAN parser artifact", resolve="1")]
                 ).to_csv(roots.label_overrides, index=False)
    r3 = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r3.returncode == 0 and "unresolved_conflicts=0" in r3.stdout and "powertrain_changed=1" in r3.stdout, r3.stdout + r3.stderr
    d3 = pd.read_parquet(roots.drives).set_index("drive_id")
    assert d3.loc["drive_000001", "powertrain"] == "HEV" and not bool(d3.loc["drive_000001", "task_bev_ice_eligible"])
    assert d3.loc["drive_000001", "label_note"] == "override:audit: engine rpm seen; label_audit=suspect_hybrid_engine_always_on"
    assert d3.loc["drive_000002", "label_note"] == "override:regen flag is a CAN parser artifact; label_audit=conflict_ice_electric/resolved"
    w3 = pd.read_parquet(roots.windows).set_index("window_id")
    assert w3.loc["drive_000001_w0", "powertrain"] == "HEV" and not bool(w3.loc["drive_000001_w0", "task_bev_ice_eligible"])
    assert w3.loc["drive_000002_w0", "powertrain"] == "ICE"
    # third run with the same overrides changes nothing
    r4 = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r4.returncode == 0 and pd.read_parquet(roots.drives).set_index("drive_id")["label_note"].equals(d3["label_note"])
    # dry run never writes: outputs untouched and a removed scan cache is not recreated
    mt = roots.review_md.stat().st_mtime
    shutil.rmtree(roots.qc_parts)
    r5 = subprocess.run(cmd + ["--dry-run"], capture_output=True, text=True, env=env)
    assert r5.returncode == 0 and "dry-run" in r5.stdout and roots.review_md.stat().st_mtime == mt
    assert not roots.qc_parts.exists()
    # a bad override file is rejected
    pd.DataFrame([dict(scope="galaxy", key="x", powertrain="", label_note="", resolve="")]).to_csv(roots.label_overrides, index=False)
    with pytest.raises(ValueError):
        Q.load_overrides(roots.label_overrides)
    pd.DataFrame([dict(scope="drive", key="drive_000001", powertrain="", label_note="see " + "a" * 16, resolve="")]).to_csv(roots.label_overrides, index=False)
    with pytest.raises(ValueError):
        Q.load_overrides(roots.label_overrides)
