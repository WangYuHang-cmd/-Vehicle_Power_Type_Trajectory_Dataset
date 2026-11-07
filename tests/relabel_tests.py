"""S5 --relabel: label columns of an existing drives index refreshed from routes_labels without touching drive tables."""
from __future__ import annotations

import pandas as pd

from vtt import config as C
from vtt import schema as S
from vtt.drives import relabel_index


def _index(n: int) -> pd.DataFrame:
    cols = S.names(S.DRIVES_INDEX_COLUMNS)
    idx = pd.DataFrame([{c: None for c in cols} for _ in range(n)])
    idx["vehicle_id"] = [f"vehicle_{i + 1:04d}" for i in range(n)]
    idx["drive_id"] = [f"drive_{i + 1:06d}" for i in range(n)]
    idx["software_era"] = ["0.9"] * n
    idx["split_device"] = ["train"] * n
    idx["task_size_eligible"] = [True] * n
    return idx


def test_relabel_index_refreshes_label_columns_and_resets_splits():
    idx = _index(2)
    names = pd.DataFrame({"route_uid": ["r1", "r2"], "drive_id": ["drive_000001", "drive_000002"]})
    labels = pd.DataFrame([dict(route_uid="r1", brand="B", model="M", platform="P", body="suv", size_code="C", wheelbase_mm=2700.0,
                                track_front_mm=1600.0, track_rear_mm=1600.0, footprint_m2=4.32, curb_mass_kg=1500.0,
                                size_class="SMALL", powertrain="BEV", powertrain_raw="BEV", label_source="fingerprint",
                                label_note=None, region="europe", software_era="0.10")])
    new, counts = relabel_index(idx, names, labels)
    assert counts == dict(drives=2, relabelled=1, no_label=1)
    r = new.set_index("drive_id").loc["drive_000001"]
    assert r.size_class == "SMALL" and r.powertrain == "BEV" and r.model == "M" and r.footprint_m2 == 4.32
    assert bool(r.task_size_eligible) and bool(r.task_bev_ice_eligible) and r.software_era == "0.10"
    r2 = new.set_index("drive_id").loc["drive_000002"]
    assert r2.size_class is None and not bool(r2.task_size_eligible) and not bool(r2.task_bev_ice_eligible)
    assert r2.software_era == "0.9"                      # kept from the index when the labels have no row
    assert new["split_device"].isna().all() and new["split_platform"].isna().all() and new["low_support_cell"].isna().all()
    assert list(new.columns) == S.names(S.DRIVES_INDEX_COLUMNS)


def test_relabel_index_unknown_size_is_not_eligible():
    idx = _index(1)
    names = pd.DataFrame({"route_uid": ["r1"], "drive_id": ["drive_000001"]})
    labels = pd.DataFrame([dict(route_uid="r1", brand="B", model="M", platform="P", body="suv", size_code="C",
                                size_class=None, powertrain="HEV", powertrain_raw="HEV", label_source="directory", region="asia")])
    new, counts = relabel_index(idx, names, labels)
    assert counts["relabelled"] == 1
    r = new.iloc[0]
    assert r.size_class is None and not bool(r.task_size_eligible)
    assert r.powertrain == "HEV" and not bool(r.task_bev_ice_eligible)       # HEV is outside the BEV-vs-ICE task
    assert C.MAIN_POWERTRAIN_TASK == ("BEV", "ICE")
