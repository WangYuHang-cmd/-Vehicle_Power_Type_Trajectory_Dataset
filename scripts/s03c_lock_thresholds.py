#!/usr/bin/env python3
"""S3c: write docs/SIZE_CLASSES.md for a pair of footprint thresholds and (optionally) lock them into vtt/config.py.

reads   build/routes_labels.parquet, build/drive_names.parquet, build/index/drives.parquet (hours per route)
writes  docs/SIZE_CLASSES.md (thresholds, rule, size x powertrain cell table, platforms near each threshold, models
        whose generations fall in two classes, every model per class) and, with --write-config, sets
        config.FOOTPRINT_THRESHOLDS_M2 and config.SIZE_CLASSES_SHA256 (sha256 of the written document).

The document carries model names, platform names, counts and hours only. Prints the thresholds, the cell table and the
sha256. The lock is what s08_splits.py checks; changing the thresholds later means re-running s03_labels -> s05 --relabel
-> s06 -> s07b -> s08 and every experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from vtt.labels import cut_size_classes  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PT_ORDER = list(C.POWERTRAIN4) + ["null"]
SIZE_ORDER = list(C.SIZE_CLASSES)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", required=True, help="t1,t2 in m^2 (SMALL < t1 <= MID < t2 <= LARGE)")
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--rule", type=Path, default=None, help="markdown file with the rationale, inserted verbatim")
    ap.add_argument("--out", type=Path, default=REPO / C.SIZE_CLASSES_DOC)
    ap.add_argument("--near", type=float, default=0.05, help="list platforms within +-this of a threshold (m^2)")
    ap.add_argument("--write-config", action="store_true", help="set FOOTPRINT_THRESHOLDS_M2 and SIZE_CLASSES_SHA256 in vtt/config.py")
    ap.add_argument("--locked-on", default=None, help="date text for the document (default: today, as '6 Nov 2026')")
    return ap.parse_args(argv)


def load(root: Path) -> pd.DataFrame:
    lab = pd.read_parquet(root / "build" / "routes_labels.parquet")
    names = pd.read_parquet(root / "build" / "drive_names.parquet")[["route_uid", "drive_id", "vehicle_id", "group_id"]]
    dr = pd.read_parquet(root / "build" / "index" / "drives.parquet", columns=["drive_id", "duration_s"])
    df = lab.merge(names, on="route_uid", how="left").merge(dr, on="drive_id", how="left")
    df["hours"] = pd.to_numeric(df["duration_s"], errors="coerce").fillna(0.0) / 3600.0
    df["footprint_m2"] = pd.to_numeric(df["footprint_m2"], errors="coerce")
    return df


def cell_table(df: pd.DataFrame, thr: tuple[float, float]) -> pd.DataFrame:
    d = df.copy()
    d["size"] = cut_size_classes(d["footprint_m2"], thr).fillna("null")
    d["pt"] = d["powertrain"].where(d["powertrain"].notna(), "null")
    g = d.groupby(["size", "pt"])
    out = pd.DataFrame({"routes": g.size(), "hours": g["hours"].sum(), "models": g["model"].nunique(),
                        "platforms": g["platform"].nunique(), "groups": g["group_id"].nunique(),
                        "vehicles": g["vehicle_id"].nunique()}).reset_index()
    idx = pd.MultiIndex.from_product([SIZE_ORDER + ["null"], PT_ORDER], names=["size", "pt"])
    out = out.set_index(["size", "pt"]).reindex(idx).fillna(0).reset_index()
    for c in ("routes", "models", "platforms", "groups", "vehicles"):
        out[c] = out[c].astype(int)
    return out


def render_cells(cells: pd.DataFrame) -> list[str]:
    L = ["| size \\ powertrain | " + " | ".join(PT_ORDER) + " | total |", "|---|" + "---|" * (len(PT_ORDER) + 1)]
    for s in SIZE_ORDER + ["null"]:
        row = cells[cells["size"] == s]
        parts = []
        for p in PT_ORDER:
            r = row[row["pt"] == p].iloc[0]
            parts.append(f"{r.hours:.0f} h / {int(r.models)} m / {int(r.groups)} g" if r.routes else "-")
        parts.append(f"{row.hours.sum():.0f} h / {int(row.routes.sum())} routes")
        L.append(f"| {s} | " + " | ".join(parts) + " |")
    L.append("")
    L.append("cells: hours h / distinct car models m / identity groups g (devices after merging); null = label unknown")
    return L


def platform_table(df: pd.DataFrame) -> pd.DataFrame:
    k = df.dropna(subset=["footprint_m2"])
    k = k[k["model"] != "Unknown"]
    g = k.groupby("platform")
    return pd.DataFrame({"footprint_m2": g["footprint_m2"].median(), "hours": g["hours"].sum(),
                         "model": g["model"].agg(lambda s: s.mode().iloc[0]),
                         "powertrains": g["powertrain"].agg(lambda s: "/".join(sorted(set(s.dropna()))) or "-")}).reset_index()


def main(argv=None) -> int:
    a = parse_args(argv)
    t1, t2 = sorted(float(x) for x in a.thresholds.split(","))
    thr = (t1, t2)
    df = load(a.data_root)
    cells = cell_table(df, thr)
    plats = platform_table(df)
    plats["size_class"] = cut_size_classes(plats["footprint_m2"], thr)
    today = a.locked_on or date.today().strftime("%-d %b %Y")

    L = ["# Size classes (locked thresholds)", "",
         f"locked on {today}, before any classifier was trained on the data.", "",
         "## Definition", "",
         "footprint = wheelbase [m] x (front track + rear track) / 2 [m], per vehicle platform (generation) as recorded in",
         "`vtt/specs_data/platform_specs.csv`. A route's `size_class` is cut from its platform's footprint:", "",
         f"| class | footprint range (m^2) |", "|---|---|",
         f"| SMALL | footprint < {t1:.2f} |", f"| MID | {t1:.2f} <= footprint < {t2:.2f} |", f"| LARGE | footprint >= {t2:.2f} |", "",
         f"`config.FOOTPRINT_THRESHOLDS_M2 = ({t1:.2f}, {t2:.2f})`; `config.SIZE_CLASSES_SHA256` is the sha256 of this file and",
         "`scripts/s08_splits.py` refuses to run when it differs. Curb mass and body style are published as separate columns",
         "and take no part in the class.", ""]
    if a.rule and Path(a.rule).exists():
        L += ["## Rationale", "", Path(a.rule).read_text().strip(), ""]
    L += ["## Cell table (routes with a drive table)", ""] + render_cells(cells)
    tot = cells.hours.sum()
    L += ["", "| class | hours | share |", "|---|---|---|"]
    for s in SIZE_ORDER + ["null"]:
        h = cells[cells["size"] == s].hours.sum()
        L.append(f"| {s} | {h:.0f} | {100 * h / tot:.1f} % |")
    for t in thr:
        near = plats[(plats["footprint_m2"] - t).abs() <= a.near + 1e-9].sort_values("footprint_m2")
        L += ["", f"## Platforms within +-{a.near:.2f} m^2 of {t:.2f}", "", "| platform | model | powertrains | footprint | hours | class |", "|---|---|---|---|---|---|"]
        for r in near.itertuples():
            L.append(f"| {r.platform} | {r.model} | {r.powertrains} | {r.footprint_m2:.3f} | {r.hours:.1f} | {r.size_class} |")
        if not len(near):
            L.append("| none | | | | | |")
    # models whose platforms fall in two classes
    k = plats.dropna(subset=["size_class"])
    strad = k.groupby("model")["size_class"].nunique()
    strad = strad[strad > 1].index.tolist()
    L += ["", "## Models whose generations fall in two classes", ""]
    if strad:
        L += ["| model | class | platforms | hours |", "|---|---|---|---|"]
        for m in sorted(strad):
            sub = k[k["model"] == m]
            for cls, g in sub.groupby("size_class"):
                L.append(f"| {m} | {cls} | {', '.join(sorted(g.platform))} | {g.hours.sum():.1f} |")
    else:
        L.append("none")
    L += ["", "## Models per class (hours with a drive table)", ""]
    for s in SIZE_ORDER:
        sub = k[k["size_class"] == s].groupby("model").agg(hours=("hours", "sum"), footprint=("footprint_m2", "median")).sort_values("footprint")
        L += [f"### {s} ({sub.hours.sum():.0f} h, {len(sub)} models)", "",
              ", ".join(f"{m} ({r.footprint:.2f}, {r.hours:.0f} h)" for m, r in sub.iterrows()), ""]
    L += ["## Changing the thresholds", "",
          "Edit the thresholds, re-run `scripts/s03c_lock_thresholds.py --write-config`, then `s03_labels.py`,",
          "`s05_build_drives.py --relabel`, `s06_windows.py`, `s07b_apply_exclusions.py`, `s08_splits.py` and every experiment.", ""]
    text = "\n".join(L)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(f"thresholds: SMALL < {t1:.2f} <= MID < {t2:.2f} <= LARGE")
    print("\n".join(render_cells(cells)))
    print(f"wrote {a.out.relative_to(REPO) if a.out.is_relative_to(REPO) else a.out}  sha256={sha}")
    if a.write_config:
        cfg = REPO / "vtt" / "config.py"
        src = cfg.read_text()
        src, n1 = re.subn(r"^FOOTPRINT_THRESHOLDS_M2: tuple\[float, float\] \| None = .*$",
                          f"FOOTPRINT_THRESHOLDS_M2: tuple[float, float] | None = ({t1:.2f}, {t2:.2f})", src, flags=re.M)
        src, n2 = re.subn(r"^SIZE_CLASSES_SHA256: str \| None = .*$", f'SIZE_CLASSES_SHA256: str | None = "{sha}"', src, flags=re.M)
        if n1 != 1 or n2 != 1:
            print("config.py lines not found; nothing written")
            return 2
        cfg.write_text(src)
        print("config.py updated (FOOTPRINT_THRESHOLDS_M2, SIZE_CLASSES_SHA256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
