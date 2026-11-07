#!/usr/bin/env python3
"""S2d: acceptance of the extraction (vtt.extract_accept) from the S2c tables.

Per software era: share of full qlog segments with >= 500 carState and >= 150 radarState rows, unknown-union histogram,
extractor_version check; coverage against the chunk lists (else uid_map); failures from work/logs/s02_failures.csv (counts
and status types only). Writes build/qc/s02_accept.csv and .md. Exit 2 when any era has < 99 % good segments, when more than
0.5 % of the expected routes are missing, when a route_meta carries another extractor version, or when the S2c tables are
stale (extracted routes not in routes_meta: re-run s02c first).

  source env.sh && $PY scripts/s02d_accept.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from vtt import config as C
from vtt import extract_accept as EA
from vtt import segments as SG
from vtt.paths import P
from vtt.private import write_text_atomic
from vtt.reroot import reroot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(C.DATA_ROOT))
    ap.add_argument("--tables-dir", default=None, help="default <data-root>/work/tables")
    ap.add_argument("--signals-dir", default=None, help="default <data-root>/work/signals")
    ap.add_argument("--uid-map", default=None, help="default <data-root>/private/uid_map.parquet")
    ap.add_argument("--chunks-dir", default=None, help="default <data-root>/build/chunks")
    ap.add_argument("--failures", default=None, help="default <data-root>/work/logs/s02_failures.csv")
    ap.add_argument("--out-dir", default=None, help="default <data-root>/build/qc")
    a = ap.parse_args(argv)
    tables = Path(a.tables_dir) if a.tables_dir else reroot(P.SEGMENTS, a.data_root).parent
    signals = Path(a.signals_dir) if a.signals_dir else reroot(P.SIGNALS, a.data_root)
    uid_map = Path(a.uid_map) if a.uid_map else reroot(P.UID_MAP, a.data_root)
    chunks = Path(a.chunks_dir) if a.chunks_dir else reroot(P.CHUNKS, a.data_root)
    failures = Path(a.failures) if a.failures else reroot(P.S02_FAILURES, a.data_root)
    out_dir = Path(a.out_dir) if a.out_dir else reroot(P.QC, a.data_root)
    needed = [tables / P.SEGMENTS.name, tables / SG.SEGMENT_COUNTS_FILE, tables / SG.ROUTES_META_FILE]
    if not all(p.exists() for p in needed):
        print("S2c tables missing: run scripts/s02c_segments_table.py first", flush=True)
        return 1
    segments, counts, routes_meta = (pd.read_parquet(p) for p in needed)

    extracted = set(SG.list_extracted(signals))
    in_tables = set(routes_meta["route_uid"].astype(str))
    stale = extracted - in_tables
    expected, source = EA.expected_uids(chunks, uid_map)
    missing = expected - extracted
    uidmap_missing = 0
    if uid_map.exists():
        uidmap_missing = len(set(pd.read_parquet(uid_map, columns=["route_uid"])["route_uid"].astype(str)) - extracted)
    coverage = {"n_expected": len(expected), "source": source, "n_extracted": len(extracted), "n_missing": len(missing),
                "n_unexpected": len(extracted - expected) if expected else 0, "n_uidmap_missing": uidmap_missing}
    fails = EA.failures_table(failures, extracted)
    era = EA.era_table(segments, counts, routes_meta)
    ok, reasons = EA.evaluate(era, len(expected), len(missing))
    if stale:
        ok, reasons = False, reasons + [f"S2c tables stale: {len(stale)} extracted routes not in routes_meta (re-run s02c)"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(era.to_csv(index=False, float_format="%.6g"), out_dir / "s02_accept.csv")
    write_text_atomic(EA.render_md(era, ok, reasons, coverage, fails), out_dir / "s02_accept.md")
    print("coverage: " + " ".join(f"{k}={v}" for k, v in coverage.items()), flush=True)
    print(f"failures: rows={fails['n_rows']} routes={fails['n_uids']} unresolved={fails['n_unresolved']} statuses={fails['status_hist']}", flush=True)
    show = ["software_era", "n_routes", "n_segments", "n_full", "n_good", "good_frac", "n_low_carstate", "n_low_radar", "unk_median", "unk_max"]
    print(era[show].to_string(index=False), flush=True)
    print(("ACCEPTED" if ok else "REJECTED: " + "; ".join(reasons)), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
