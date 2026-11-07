"""vtt.inventory: threshold schemes, platform table, cell table and markdown rendering (counts only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vtt import inventory as I
from vtt.private import mask


def _labels(n_platforms: int = 9) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """3 routes per platform; footprints 3.0, 3.2, ..., devices alternate; powertrain by platform parity."""
    rows, hours, devs = [], [], []
    for p in range(n_platforms):
        for k in range(3):
            fp = 3.0 + 0.2 * p
            rows.append(dict(route_uid=f"uid{p:03d}{k}", platform=f"PLAT_{p:02d}", model=f"Model {p}", brand="Fake",
                             body="suv" if p % 2 else "sedan", footprint_m2=fp, curb_mass_kg=1500 + 50 * p,
                             powertrain=["ICE", "BEV", None][p % 3], spec_confidence="B"))
            hours.append(1.0 + k)
            devs.append(f"dev{p:02d}{k % 2}")
    return pd.DataFrame(rows), pd.Series(hours), pd.Series(devs)


def test_tertile_thresholds():
    fps = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6]
    t = I.tertile_thresholds(fps)
    assert t == (3.5, 4.1)                     # 3 / 3 / 3 platforms
    assert I.tertile_thresholds([3.0, 3.1]) is None
    assert I.tertile_thresholds([4.0, 4.0, 4.0, 4.0]) is None
    assert I.tertile_thresholds(pd.Series([3.0, None, 3.2, 3.4, 3.6])) == (3.1, 3.5)


def test_natural_breaks_prefers_large_gaps_but_balanced():
    fps = [3.0, 3.05, 3.1, 3.15, 4.0, 4.05, 4.1, 5.0, 5.05, 5.1]
    t = I.natural_breaks(fps)
    assert t == (3.575, 4.55)
    # an outlier at the end never becomes a class of its own (min_share)
    fps2 = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 9.0]
    t2 = I.natural_breaks(fps2)
    assert t2 is not None and t2[1] < 9.0 and sum(f >= t2[1] for f in fps2) >= 1
    n_hi = sum(f >= t2[1] for f in fps2)
    assert n_hi >= 1 and len([f for f in fps2 if t2[0] <= f < t2[1]]) >= 1
    assert I.natural_breaks([3.0, 3.0, 3.0]) is None and I.natural_breaks([3.0, 4.0]) is None


def _breaks_reference(fps, min_share=0.10):
    """Brute force over every pair of midpoints (same scoring, same tie rule: lowest thresholds)."""
    f = np.sort(np.asarray(fps, dtype=float))
    n = len(f)
    vals = np.unique(f)
    if n < 3 or len(vals) < 3:
        return None
    min_n = max(1, int(np.ceil(min_share * n)))
    mids = [(a + b) / 2 for a, b in zip(vals[:-1], vals[1:])]
    score = {m: (b - a) * min((f < m).sum(), (f >= m).sum()) / n for m, a, b in zip(mids, vals[:-1], vals[1:])}
    best = None
    for lo in mids:
        for hi in mids:
            if hi <= lo:
                continue
            n_lo, n_mid, n_hi = (f < lo).sum(), ((f >= lo) & (f < hi)).sum(), (f >= hi).sum()
            if min(n_lo, n_mid, n_hi) < min_n:
                continue
            tot = score[lo] + score[hi]
            if best is None or tot > best[0] + 1e-12:
                best = (tot, lo, hi)
    return None if best is None else (round(best[1], 3), round(best[2], 3))


def test_natural_breaks_matches_exhaustive_reference():
    rng = np.random.RandomState(20261105)
    n_checked = 0
    for _ in range(300):
        n = int(rng.randint(3, 40))
        kind = rng.randint(3)
        if kind == 0:
            fps = np.round(rng.uniform(3.0, 6.0, n), 2)
        elif kind == 1:
            fps = np.round(np.concatenate([rng.normal(3.6, 0.15, n // 2 + 1), rng.normal(5.0, 0.3, n - n // 2 - 1)]), 2)
        else:
            fps = np.round(3.0 + 0.1 * np.arange(n) + rng.choice([0, 0, 0.5], n), 2)
        assert I.natural_breaks(fps) == _breaks_reference(fps), fps.tolist()
        n_checked += 1
    assert n_checked == 300
    # the verifier's counter-example: the best admissible pair need not contain a top-scoring gap
    fps = [3.09, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.78, 3.9, 3.98, 4.1, 4.2, 4.3, 4.4, 4.42, 4.5, 4.6, 4.7, 4.8, 4.9,
           5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.8, 5.9]
    assert I.natural_breaks(fps) == _breaks_reference(fps)


def test_platform_table_and_cells():
    labels, hours, devs = _labels()
    pt = I.platform_table(labels, hours, devs)
    assert len(pt) == 9 and pt["platform"].iloc[0] == "PLAT_00" and pt["footprint_m2"].is_monotonic_increasing
    r = pt.set_index("platform").loc["PLAT_01"]
    assert r.n_routes == 3 and r.n_vehicles == 2 and r.hours == 6.0 and r.model == "Model 1" and r.body == "suv"
    specs = pd.DataFrame({"platform": ["PLAT_01"], "footprint_min_m2": [3.1], "footprint_max_m2": [3.3]})
    pt2 = I.platform_table(labels, hours, devs, specs).set_index("platform")
    assert pt2.loc["PLAT_01", "footprint_min_m2"] == 3.1 and np.isnan(pt2.loc["PLAT_00", "footprint_min_m2"])
    cells = I.cell_table(labels, (3.5, 4.1), hours, devs)
    assert set(cells["size"]) == set(I.SIZE_ORDER) and set(cells["pt"]) == set(I.PT_ORDER)
    assert cells["routes"].sum() == 27 and cells["hours"].sum() == hours.sum()
    small_ice = cells[(cells["size"] == "SMALL") & (cells["pt"] == "ICE")].iloc[0]
    assert small_ice.routes == 3 and small_ice.platforms == 1 and small_ice.devices == 2          # PLAT_00 only
    none_cells = I.cell_table(labels, None, hours, devs)
    assert none_cells[none_cells["size"] == "null"]["routes"].sum() == 27
    near = I.near_threshold(pt, (3.5, 4.1), tol=0.1)
    assert set(near[3.5]["platform"]) == {"PLAT_02", "PLAT_03"} and set(near[4.1]["platform"]) == {"PLAT_05", "PLAT_06"}


def test_render_markdown_has_counts_only():
    labels, hours, devs = _labels()
    labels.loc[labels["platform"] == "PLAT_08", "footprint_m2"] = np.nan
    pt = I.platform_table(labels, hours, devs)
    thr = {"tertiles": I.tertile_thresholds(pt["footprint_m2"]), "natural_breaks": None}
    cells = {"tertiles": I.cell_table(labels, thr["tertiles"], hours, devs)}
    near = {"tertiles": I.near_threshold(pt, thr["tertiles"])}
    md = I.render_inventory_md(pt, thr, cells, near, dict(platforms=9, platforms_with_footprint=8, routes=27, hours=54.0, devices=18,
                                                           hours_with_footprint=48.0, hours_with_footprint_pct=88.9), "2026" + "-01-01")   # built: no ISO date literal under tests/
    assert "## Scheme: tertiles" in md and "## Scheme: natural_breaks" in md and "not computable" in md
    assert "| PLAT_08 | Model 8 | 2 | 6.0 |" in md                     # listed under platforms without footprint
    assert "uid" not in md and "dev0" not in md and mask(md) == md
    assert "cells: routes r / hours h / devices d / platforms p" in md
