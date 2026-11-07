"""S2b tests: the label-audit probe on synthetic qlogs (ICE / BEV / HEV profiles, missing dir, cut file), the NaN policy for
fields a writer lacks, the private table schema round trip and the resumable script."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests import synth_qlog as SQ
from vtt import audit_probe as AP
from vtt.tocf_bridge import tocf

R = Path(__file__).resolve().parents[1]
K = tocf("capnp_io")


def load_vtt_script(name: str):
    spec = importlib.util.spec_from_file_location(name, R / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("s02b")
    raw = root / "raw"
    spec = {"ice": dict(profile="ice", n_seg=4), "bev": dict(profile="bev", n_seg=2), "hev": dict(profile="hev", n_seg=1, last_seg_s=30.0),
            "cut": dict(profile="ice", n_seg=2, truncate_seg=1, zst_segs=(0,))}
    dirs, rows = {}, []
    for i, (name, kw) in enumerate(spec.items(), start=1):
        dongle, route = SQ.fake_dongle(i), SQ.fake_route(i)
        dirs[name] = raw / f"CAR_{name.upper()}" / dongle / route
        SQ.write_route(dirs[name], dongle, f"CAR_{name.upper()}", v_mps=12.0, **kw)
        rows.append(dict(car_dir=f"CAR_{name.upper()}", dongle_dir=dongle, dongle_log=dongle, route_id=route))
    dirs["missing"] = raw / "CAR_M" / SQ.fake_dongle(8) / SQ.fake_route(8)
    return {"root": root, "raw": raw, "dirs": dirs, "rows": rows}


def test_probe_profiles(tree):
    ice = AP.probe_route(tree["dirs"]["ice"])
    assert ice["n_segments_read"] == 3 and ice["n_samples"] == 1800 and ice["err"] == ""
    assert json.loads(ice["ecus"]) == ["engine", "eps", "fwdRadar", "transmission"]
    assert ice["has_transmission_ecu"] is True and ice["has_hybrid_ecu"] is False and ice["transmission"] == "automatic"
    assert abs(ice["mass_kg"] - 1600) < 1e-3 and abs(ice["wheelbase_m"] - 2.7) < 1e-5 and abs(ice["steer_ratio"] - 15) < 1e-6
    assert ice["engine_rpm_pos_frac"] == 1.0 and ice["engine_off_moving_frac"] == 0.0
    assert ice["regen_any"] is False and ice["charging_any"] is False and abs(ice["fuel_gauge_med"] - 0.6) < 1e-6
    assert json.loads(ice["gear_values"]) == ["drive", "park"] and ice["yaw_rate_nonzero_frac"] == 0.75 and ice["eps_torque_nonzero_frac"] == 0.5
    bev = AP.probe_route(tree["dirs"]["bev"])
    assert bev["n_segments_read"] == 2 and bev["engine_rpm_pos_frac"] == 0.0 and bev["engine_off_moving_frac"] == 1.0
    assert bev["regen_any"] is True and bev["transmission"] == "direct" and bev["has_transmission_ecu"] is False
    assert json.loads(bev["gear_values"]) == ["drive"]
    hev = AP.probe_route(tree["dirs"]["hev"])
    assert hev["has_hybrid_ecu"] is True and hev["transmission"] == "cvt" and hev["n_samples"] == 300


def test_probe_missing_dir_cut_file_and_max_segs(tree):
    m = AP.probe_route(tree["dirs"]["missing"])
    assert m["err"] == "missing_dir" and m["n_segments_read"] == 0 and m["n_samples"] == 0 and m["ecus"] is None
    assert np.isnan(m["mass_kg"]) and m["regen_any"] is None
    cut = AP.probe_route(tree["dirs"]["cut"])
    assert cut["n_segments_read"] == 2 and 600 < cut["n_samples"] < 1200 and cut["err"] == ""   # the cut file yields its prefix
    one = AP.probe_route(tree["dirs"]["ice"], max_segs=1)
    assert one["n_segments_read"] == 1 and one["n_samples"] == 600
    empty = tree["root"] / "empty_route"
    empty.mkdir()
    assert AP.probe_route(empty)["err"] == "no_logs"


def test_missing_fields_are_nan():
    """A writer whose carState lacks the newer fields: stats stay NaN / null instead of capnp defaults."""
    samples = [{"vEgo": 10.0, "gearShifter": "drive", "yawRate": 0.0, "steeringTorqueEps": 1.0},
               {"vEgo": 0.0, "gearShifter": "park", "yawRate": 0.1, "steeringTorqueEps": 0.0}]
    present = {"vEgo", "gearShifter", "yawRate", "steeringTorqueEps"}
    st = AP.carstate_stats(samples, present)
    assert np.isnan(st["engine_rpm_pos_frac"]) and np.isnan(st["engine_off_moving_frac"]) and np.isnan(st["fuel_gauge_med"])
    assert st["regen_any"] is None and st["charging_any"] is None
    assert json.loads(st["gear_values"]) == ["drive", "park"] and st["yaw_rate_nonzero_frac"] == 0.5 and st["eps_torque_nonzero_frac"] == 0.5
    assert st["n_samples"] == 2
    assert AP.carstate_stats([], set())["n_samples"] == 0
    # no moving sample -> engine_off_moving_frac NaN even when engineRpm is present
    st = AP.carstate_stats([{"vEgo": 1.0, "engineRpm": 0.0}], {"vEgo", "engineRpm"})
    assert st["engine_rpm_pos_frac"] == 0.0 and np.isnan(st["engine_off_moving_frac"])
    # the parse schema's own writer has every probed field in its layout
    ev, cs = SQ.new_event(1, "carState")
    assert AP.carstate_present(ev.as_reader().carState) == set(AP.CARSTATE_FIELDS)
    ev, cp = SQ.new_event(2, "carParams")
    cp.mass = 1234.0
    f = AP.carparams_fields(ev.as_reader().carParams)
    assert f["mass_kg"] == 1234.0 and f["ecus"] == "[]" and f["has_hybrid_ecu"] is False and f["transmission"] == "unknown"


def test_table_roundtrip_and_merge(tmp_path):
    rows = [dict(AP.empty_row("u1", "missing_dir")), dict(AP.empty_row("u2"), has_hybrid_ecu=True, regen_any=False, mass_kg=1.5, n_samples=3)]
    df = AP.audit_frame(rows)
    assert list(df.columns) == AP.AUDIT_COLUMNS
    p = AP.write_audit(df, tmp_path / "private" / "route_audit.parquet")
    back = pd.read_parquet(p)
    assert list(back.columns) == AP.AUDIT_COLUMNS and len(back) == 2
    assert back.set_index("route_uid").at["u2", "has_hybrid_ecu"] == True and pd.isna(back.set_index("route_uid").at["u1", "has_hybrid_ecu"])  # noqa: E712
    assert str(back["n_samples"].dtype) == "int64" and back["err"].tolist() == ["missing_dir", ""]
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    merged = AP.merge_audit(back, AP.audit_frame([dict(AP.empty_row("u1"), n_samples=9)]))
    assert merged.set_index("route_uid").at["u1", "n_samples"] == 9 and len(merged) == 2
    assert len(AP.read_audit(tmp_path / "nothing.parquet")) == 0


def test_script_resume_and_errors(tree, tmp_path):
    root = tmp_path / "data"
    priv = root / "private"
    SQ.write_private(priv, tree["rows"])                           # the four readable routes
    s02b = load_vtt_script("s02b_audit_probe")
    args = ["--data-root", str(root), "--raw-root", str(tree["raw"]), "--workers", "2", "--batch", "2", "--nice", "0"]
    assert s02b.main(args + ["--dry-run"]) == 0
    assert not (priv / "route_audit.parquet").exists()
    assert s02b.main(args) == 0
    t = pd.read_parquet(priv / "route_audit.parquet")
    assert len(t) == 4 and (t["err"] == "").all() and (t["n_samples"] > 0).all()
    assert s02b.main(args + ["--limit", "1"]) == 0 and len(pd.read_parquet(priv / "route_audit.parquet")) == 4   # resume: nothing to do
    # a route whose raw dir is absent: probed with err, pushes the error share over the limit -> exit 2; --retry-errors re-queues it
    d = tree["dirs"]["missing"]
    SQ.write_private(priv, tree["rows"] + [dict(car_dir=d.parents[1].name, dongle_dir=d.parent.name, dongle_log=d.parent.name, route_id=d.name)])
    assert s02b.main(args) == 2
    t = pd.read_parquet(priv / "route_audit.parquet")
    assert len(t) == 5 and (t["err"] == "missing_dir").sum() == 1
    assert s02b.main(args + ["--dry-run"]) == 0 and s02b.main(args + ["--retry-errors", "--workers", "1"]) == 2
    assert len(pd.read_parquet(priv / "route_audit.parquet")) == 5
    assert s02b.main(["--data-root", str(tmp_path / "nowhere"), "--raw-root", str(tree["raw"])]) == 1
