#!/usr/bin/env python3
"""S2c: rebuild work/tables/segments.parquet (tocf.schema.SEGMENTS) from every work/signals/<uid>/route_meta.json, with
rate_carState / rate_radarState / rate_dm recomputed from the parquets, plus the two side tables the acceptance reads
(work/tables/segment_counts.parquet, work/tables/routes_meta.parquet; vtt.segments). Full rebuild, atomic writes, counts only.

  source env.sh && nice -n 15 ionice -c3 $PY scripts/s02c_segments_table.py --workers 8
  $PY scripts/s02c_segments_table.py --signals-dir <other extraction> --routes <uids.csv> --tables-dir $D/tmp/x   # ad-hoc check
Exit 2 when any extracted route could not be read (its row is left out of the tables).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from vtt import config as C
from vtt import segments as SG
from vtt.paths import P
from vtt.private import mask, write_parquet_atomic
from vtt.reroot import reroot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(C.DATA_ROOT))
    ap.add_argument("--signals-dir", default=None, help="default <data-root>/work/signals")
    ap.add_argument("--tables-dir", default=None, help="default <data-root>/work/tables")
    ap.add_argument("--routes", default=None, help="CSV with a route_uid column: rebuild only these (default: every extracted route)")
    ap.add_argument("--workers", type=int, default=C.EXTRACT_WORKERS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    signals = Path(a.signals_dir) if a.signals_dir else reroot(P.SIGNALS, a.data_root)
    tables = Path(a.tables_dir) if a.tables_dir else reroot(P.SEGMENTS, a.data_root).parent
    uids = SG.list_extracted(signals)
    if a.routes:
        wanted = set(pd.read_csv(a.routes, dtype=str)["route_uid"].astype(str).str.strip())
        uids = [u for u in uids if u in wanted]
        print(f"routes requested {len(wanted)}, extracted among them {len(uids)}", flush=True)
    print(f"extracted routes {len(uids)} under signals dir; tables -> {mask(str(tables))}", flush=True)
    if a.dry_run:
        return 0
    res = SG.rebuild(signals, uids, workers=max(1, a.workers))
    tables.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(res["segments"], tables / P.SEGMENTS.name)
    write_parquet_atomic(res["counts"], tables / SG.SEGMENT_COUNTS_FILE)
    write_parquet_atomic(res["routes_meta"], tables / SG.ROUTES_META_FILE)
    summ = SG.summary(res["segments"], res["routes_meta"])
    summ["unreadable_routes"] = len(res["errors"])
    if res["errors"]:
        hist: dict[str, int] = {}
        for t in res["errors"].values():
            hist[t] = hist.get(t, 0) + 1
        summ["unreadable_types"] = hist
    print(json.dumps(summ, indent=1, default=str), flush=True)
    return 2 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
