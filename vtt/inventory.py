"""Size inventory: per-platform table, candidate footprint thresholds and size x powertrain cell tables.

Units: footprint m^2, mass kg, hours h. Everything here works on counts and public labels only (no identifiers);
the device count is computed from a hashed-id column that never leaves the caller.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from vtt import config as C
from vtt.labels import cut_size_classes

PT_ORDER = list(C.POWERTRAIN4) + ["null"]
SIZE_ORDER = list(C.SIZE_CLASSES) + ["null"]
SCHEME_TERTILES = "tertiles"
SCHEME_BREAKS = "natural_breaks"


def _median(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.median()) if len(v) else float("nan")


def _mode(s: pd.Series):
    s = s.dropna()
    if not len(s):
        return None
    return s.value_counts().index[0]


def platform_table(labels: pd.DataFrame, hours: pd.Series, device_key: pd.Series | None = None,
                   specs: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per platform: model, brand, body, footprint (min/max from specs when given), curb_mass_kg,
    n_vehicles (distinct device_key), n_routes, hours, spec_confidence; sorted by footprint (unknown last).
    `hours` and `device_key` are aligned with `labels` rows."""
    df = labels.copy()
    df["_h"] = pd.to_numeric(hours, errors="coerce").fillna(0).to_numpy()
    df["_dev"] = device_key.to_numpy() if device_key is not None else np.arange(len(df))
    df["platform"] = df["platform"].where(df["platform"].notna(), "(no platform)")
    g = df.groupby("platform", sort=True)
    out = pd.DataFrame({
        "model": g["model"].agg(_mode), "brand": g["brand"].agg(_mode), "body": g["body"].agg(_mode),
        "footprint_m2": g["footprint_m2"].agg(_median),
        "curb_mass_kg": g["curb_mass_kg"].agg(_median),
        "n_vehicles": g["_dev"].nunique(), "n_routes": g.size(), "hours": g["_h"].sum(),
        "spec_confidence": g["spec_confidence"].agg(_mode),
    }).reset_index()
    out["footprint_min_m2"] = np.nan
    out["footprint_max_m2"] = np.nan
    if specs is not None and len(specs):
        sp = specs.set_index("platform")
        for c in ("footprint_min_m2", "footprint_max_m2"):
            if c in sp.columns:
                out[c] = out["platform"].map(pd.to_numeric(sp[c], errors="coerce")).to_numpy()
    out = out.sort_values(["footprint_m2", "platform"], na_position="last").reset_index(drop=True)
    return out


def tertile_thresholds(footprints) -> tuple[float, float] | None:
    """Thresholds that split the platforms (one vote each) into three equal-count groups by footprint:
    midpoints between the sorted footprints at ranks N/3 and 2N/3. None with fewer than 3 platforms."""
    f = np.sort(pd.to_numeric(pd.Series(footprints), errors="coerce").dropna().to_numpy())
    n = len(f)
    if n < 3:
        return None
    i1 = min(max(1, int(round(n / 3))), n - 2)
    i2 = min(max(i1 + 1, int(round(2 * n / 3))), n - 1)
    t1 = float((f[i1 - 1] + f[i1]) / 2)
    t2 = float((f[i2 - 1] + f[i2]) / 2)
    if t2 <= t1:
        return None
    return round(t1, 3), round(t2, 3)


def natural_breaks(footprints, min_share: float = 0.10) -> tuple[float, float] | None:
    """Two largest gaps between consecutive distinct platform footprints, each gap weighted by the smaller platform
    count on its two sides (gap x min(n_below, n_above) / N), so a break never isolates a handful of platforms:
    every class must hold at least `min_share` of the platforms. Every admissible pair of gaps is scored (the best pair
    need not contain the top-scoring gap once the min-share constraint bites); ties go to the lower thresholds.
    None when no admissible pair exists."""
    f = np.sort(pd.to_numeric(pd.Series(footprints), errors="coerce").dropna().to_numpy())
    n = len(f)
    if n < 3:
        return None
    vals = np.unique(f)
    if len(vals) < 3:
        return None
    min_n = max(1, int(math.ceil(min_share * n)))
    mids = (vals[:-1] + vals[1:]) / 2.0
    n_below = np.searchsorted(f, mids, side="left")            # platforms strictly below each midpoint
    scores = (vals[1:] - vals[:-1]) * np.minimum(n_below, n - n_below) / n
    best = None
    for i in range(len(mids)):
        if n_below[i] < min_n:
            continue
        for j in range(i + 1, len(mids)):
            n_mid, n_hi = n_below[j] - n_below[i], n - n_below[j]
            if n_mid < min_n or n_hi < min_n:
                continue
            tot = float(scores[i] + scores[j])
            if best is None or tot > best[0] + 1e-12:
                best = (tot, float(mids[i]), float(mids[j]))
    if best is None:
        return None
    return round(best[1], 3), round(best[2], 3)


def cell_table(labels: pd.DataFrame, thresholds: tuple[float, float] | None, hours: pd.Series,
               device_key: pd.Series | None = None) -> pd.DataFrame:
    """size_class x powertrain cells with routes, hours, devices, platforms (null rows / columns included)."""
    df = labels[["platform", "footprint_m2", "powertrain"]].copy()
    df["_h"] = pd.to_numeric(hours, errors="coerce").fillna(0).to_numpy()
    df["_dev"] = device_key.to_numpy() if device_key is not None else np.arange(len(df))
    df["size"] = cut_size_classes(df["footprint_m2"], thresholds).fillna("null").to_numpy() if thresholds else "null"
    df["pt"] = df["powertrain"].where(df["powertrain"].notna(), "null")
    g = df.groupby(["size", "pt"])
    cells = pd.DataFrame({"routes": g.size(), "hours": g["_h"].sum(), "devices": g["_dev"].nunique(),
                          "platforms": g["platform"].nunique()}).reset_index()
    idx = pd.MultiIndex.from_product([SIZE_ORDER, PT_ORDER], names=["size", "pt"])
    cells = cells.set_index(["size", "pt"]).reindex(idx).fillna(0).reset_index()
    for c in ("routes", "devices", "platforms"):
        cells[c] = cells[c].astype(int)
    return cells


def near_threshold(ptable: pd.DataFrame, thresholds: tuple[float, float], tol: float = 0.1) -> dict[float, pd.DataFrame]:
    """Platforms whose footprint lies within +-tol m^2 of each threshold (platform, model, footprint, hours)."""
    out = {}
    f = pd.to_numeric(ptable["footprint_m2"], errors="coerce")
    for t in thresholds:
        sel = ptable[(f - t).abs() <= tol + 1e-9][["platform", "model", "footprint_m2", "hours"]]
        out[t] = sel.sort_values("footprint_m2").reset_index(drop=True)
    return out


def _fmt(v, nd=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.{nd}f}"


def render_cells_md(cells: pd.DataFrame) -> str:
    lines = ["| size \\ powertrain | " + " | ".join(PT_ORDER) + " | total |", "|---|" + "---|" * (len(PT_ORDER) + 1)]
    for s in SIZE_ORDER:
        row = cells[cells["size"] == s]
        parts = []
        for p in PT_ORDER:
            r = row[row["pt"] == p].iloc[0]
            parts.append(f"{int(r.routes)} r / {r.hours:.0f} h / {int(r.devices)} d / {int(r.platforms)} p")
        parts.append(f"{int(row.routes.sum())} r / {row.hours.sum():.0f} h")
        lines.append(f"| {s} | " + " | ".join(parts) + " |")
    lines.append("")
    lines.append("cells: routes r / hours h / devices d / platforms p")
    return "\n".join(lines)


def render_inventory_md(ptable: pd.DataFrame, schemes: dict[str, tuple[float, float] | None], cells: dict[str, pd.DataFrame],
                        near: dict[str, dict[float, pd.DataFrame]], totals: dict, generated: str) -> str:
    """SIZE_INVENTORY.md text (counts, model names and footprints only)."""
    L = [f"# Size inventory (footprint = wheelbase x mean track, m^2)", "", f"generated {generated}", "",
         f"- platforms: {totals.get('platforms', 0)} ({totals.get('platforms_with_footprint', 0)} with footprint)",
         f"- routes: {totals.get('routes', 0)}; hours: {totals.get('hours', 0):.0f}; devices: {totals.get('devices', 0)}",
         f"- hours with a footprint: {totals.get('hours_with_footprint', 0):.0f} ({totals.get('hours_with_footprint_pct', 0):.1f} %)",
         "", "## Platforms by footprint", "",
         "| # | platform | model | brand | body | footprint | min | max | mass kg | vehicles | hours | conf |", "|---|" + "---|" * 12]
    for i, r in ptable.iterrows():
        L.append(f"| {i + 1} | {r.platform} | {r.model or ''} | {r.brand or ''} | {r.body or ''} | {_fmt(r.footprint_m2)} | "
                 f"{_fmt(r.footprint_min_m2)} | {_fmt(r.footprint_max_m2)} | {_fmt(r.curb_mass_kg, 0)} | {int(r.n_vehicles)} | "
                 f"{r.hours:.1f} | {r.spec_confidence or ''} |")
    for name, thr in schemes.items():
        L += ["", f"## Scheme: {name}", ""]
        if thr is None:
            L.append("not computable (fewer than 3 platforms with a footprint)")
            continue
        L.append(f"thresholds: SMALL < {thr[0]:.3f} <= MID < {thr[1]:.3f} <= LARGE  (m^2)")
        L += ["", render_cells_md(cells[name]), ""]
        for t, df in near[name].items():
            L.append(f"platforms within +-0.1 m^2 of {t:.3f}: " + (", ".join(
                f"{r.platform} ({r.model}, {r.footprint_m2:.2f}, {r.hours:.0f} h)" for r in df.itertuples()) or "none"))
    missing = ptable[pd.to_numeric(ptable["footprint_m2"], errors="coerce").isna()]
    L += ["", "## Platforms without footprint (fill vtt/specs_data/platform_specs.csv)", ""]
    if len(missing):
        L.append("| platform | model | vehicles | hours |")
        L.append("|---|---|---|---|")
        for r in missing.sort_values("hours", ascending=False).itertuples():
            L.append(f"| {r.platform} | {r.model or ''} | {int(r.n_vehicles)} | {r.hours:.1f} |")
    else:
        L.append("none")
    L += ["", "Definitions: tertiles = platform-count tertiles of the sorted footprints (midpoints between ranks N/3 and 2N/3); "
          "natural_breaks = the two largest gaps between consecutive platform footprints, each weighted by the smaller "
          "platform count on its sides (gap x min(n_below, n_above) / N), with every class holding >= 10 % of the platforms. "
          "The locked thresholds are a user decision recorded in docs/SIZE_CLASSES.md (stage S3c).", ""]
    return "\n".join(L)
