#!/usr/bin/env python3
"""S3a (4): build/SIZE_INVENTORY.md - platforms sorted by footprint with two candidate threshold schemes.

Reads build/routes_labels.parquet (labels), private/labels_private.parquet (device column, counted only) and
private/routes_resolved.parquet (segment counts -> hours), plus vtt/specs_data/platform_specs.csv for footprint min/max.
For each scheme (tertiles by platform count; natural breaks) prints the size x powertrain cell table and the platforms
within +-0.1 m^2 of each threshold. The report holds counts, model names and footprints only (checked with
vtt.private.mask before writing). Exit 2 when inputs are missing or an identifier-like token would be written.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from vtt import config as C
from vtt import inventory as I
from vtt import labels as L
from vtt import specs as S
from vtt.paths import P
from vtt.private import mask, write_text_atomic


def parse_thresholds(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    a, b = (float(x) for x in text.split(","))
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def load_inputs(data_root: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """labels frame, hours per row, device key per row (None when the private table is absent)."""
    labels = pd.read_parquet(L.rebase(P.ROUTES_LABELS, data_root))
    hours = pd.Series(0.0, index=labels.index)
    rr = L.rebase(P.ROUTES_RESOLVED, data_root)
    if rr.exists():
        r = pd.read_parquet(rr, columns=["route_uid", "n_segments"])
        h = L.hours_from_segments(r["n_segments"]).to_numpy()
        hours = labels["route_uid"].map(dict(zip(r["route_uid"], h))).fillna(0.0)
    dev = None
    lp = L.rebase(P.LABELS_PRIVATE, data_root)
    if lp.exists():
        pr = pd.read_parquet(lp, columns=["route_uid", "dongle_log"])
        dev = labels["route_uid"].map(dict(zip(pr["route_uid"], pr["dongle_log"]))).fillna("unknown")
    return labels, hours, dev


def build_report(labels: pd.DataFrame, hours: pd.Series, dev: pd.Series | None, specs: pd.DataFrame,
                 user_thresholds: tuple[float, float] | None = None, generated: str | None = None) -> tuple[str, dict]:
    """Markdown text and the scheme thresholds."""
    ptable = I.platform_table(labels, hours, dev, specs)
    fps = ptable.loc[ptable["platform"] != "(no platform)", "footprint_m2"]
    schemes = {I.SCHEME_TERTILES: I.tertile_thresholds(fps), I.SCHEME_BREAKS: I.natural_breaks(fps)}
    if user_thresholds:
        schemes["user"] = user_thresholds
    if C.FOOTPRINT_THRESHOLDS_M2 is not None:
        schemes["locked (config)"] = tuple(C.FOOTPRINT_THRESHOLDS_M2)
    cells = {k: I.cell_table(labels, v, hours, dev) for k, v in schemes.items() if v}
    near = {k: I.near_threshold(ptable, v) for k, v in schemes.items() if v}
    has_fp = labels["footprint_m2"].notna().to_numpy()
    tot_h = float(hours.sum())
    totals = dict(platforms=int(labels["platform"].nunique()), platforms_with_footprint=int(fps.notna().sum()),
                  routes=len(labels), hours=tot_h, devices=int(dev.nunique()) if dev is not None else 0,
                  hours_with_footprint=float(hours[has_fp].sum()),
                  hours_with_footprint_pct=100.0 * float(hours[has_fp].sum()) / tot_h if tot_h else 0.0)
    text = I.render_inventory_md(ptable, schemes, cells, near, totals, generated or date.today().isoformat())
    return text, schemes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--specs-dir", type=Path, default=S.SPECS_DIR)
    ap.add_argument("--thresholds", default=None, help="extra scheme to tabulate, e.g. 3.9,4.6 (m^2)")
    ap.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    a = ap.parse_args(argv)

    lab = L.rebase(P.ROUTES_LABELS, a.data_root)
    if not lab.exists():
        print("missing build/routes_labels.parquet: run scripts/s03_labels.py first")
        return 2
    labels, hours, dev = load_inputs(a.data_root)
    specs = S.read_platform_specs(Path(a.specs_dir) / S.PLATFORM_SPECS_CSV.name)
    text, schemes = build_report(labels, hours, dev, specs, parse_thresholds(a.thresholds))
    print(f"routes {len(labels)}, platforms {labels['platform'].nunique()}, hours {hours.sum():.0f}, "
          f"footprint known for {int(labels['footprint_m2'].notna().sum())} routes")
    for k, v in schemes.items():
        print(f"scheme {k}: " + ("n/a" if v is None else f"{v[0]:.3f} / {v[1]:.3f} m^2"))
    if mask(text) != text:
        print("ACCEPTANCE FAILED: identifier-like token in the report text")
        return 2
    if a.dry_run:
        print("dry-run: nothing written")
        return 0
    out = L.rebase(P.SIZE_INVENTORY, a.data_root)
    write_text_atomic(text, out)
    print(f"wrote {out.name} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
