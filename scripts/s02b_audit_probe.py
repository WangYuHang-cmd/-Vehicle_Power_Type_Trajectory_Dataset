#!/usr/bin/env python3
"""S2b: label-audit probe -> private/route_audit.parquet (vtt.audit_probe; contract in docs/PIPELINE.md).

For every uid_map route the first --max-segs segments' logs are read with tocf.capnp_io.iter_events: carParams ECU names /
transmission / CarSpecs numbers and carState engineRpm / regen / charging / fuelGauge / gear / yawRate / EPS-torque statistics.
Resumable (routes already in the table are skipped; --retry-errors re-probes rows with a non-empty err), multiprocessing,
batched atomic writes, workers re-niced. Reads the raw library (read-only); prints counts only. One writer per route is
assumed: the carState field layout comes from the route's first carState message, carParams from the first segment holding one.

  source env.sh && nice -n 15 ionice -c3 $PY scripts/s02b_audit_probe.py --workers 6
Exit 2 when more than 1 % of the probed routes ended in an error (missing dir, no logs, read failure).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from vtt import audit_probe as AP
from vtt import config as C
from vtt.paths import P
from vtt.private import ensure_private_root, mask
from vtt.reroot import reroot

ERR_MAX_FRAC = 0.01
_MAX_SEGS = 3


def _init(nice: int, max_segs: int):
    global _MAX_SEGS
    _MAX_SEGS = max_segs
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass


def _job(args) -> dict:
    uid, route_dir = args
    try:
        row = AP.probe_route(Path(route_dir), max_segs=_MAX_SEGS)
    except Exception as e:                      # noqa: BLE001
        row = AP.empty_row(err=f"job:{type(e).__name__}")
    row["route_uid"] = uid
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(C.DATA_ROOT))
    ap.add_argument("--raw-root", default=str(C.RAW_ROOT))
    ap.add_argument("--uid-map", default=None, help="default <data-root>/private/uid_map.parquet")
    ap.add_argument("--out", default=None, help="default <data-root>/private/route_audit.parquet")
    ap.add_argument("--routes", default=None, help="CSV with a route_uid column: probe only these")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-segs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200, help="routes per atomic write of the table")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    uid_map = Path(a.uid_map) if a.uid_map else reroot(P.UID_MAP, a.data_root)
    out = Path(a.out) if a.out else reroot(P.ROUTE_AUDIT, a.data_root)
    raw = Path(a.raw_root)
    if not uid_map.exists():
        print("uid_map missing: run S1 first", flush=True)
        return 1
    um = pd.read_parquet(uid_map)
    if a.routes:
        wanted = set(pd.read_csv(a.routes, dtype=str)["route_uid"].astype(str).str.strip())
        um = um[um["route_uid"].isin(wanted)]
    table = AP.read_audit(out)
    done = set(table["route_uid"].astype(str))
    if a.retry_errors and len(table):
        done -= set(table.loc[table["err"].astype(str) != "", "route_uid"].astype(str))
    todo = um[~um["route_uid"].isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"uid_map routes {len(um)}, already probed {len(um) - len(todo)}, to probe {len(todo)}", flush=True)
    if a.dry_run or not len(todo):
        return 0
    ensure_private_root(out.parent)
    jobs = [(str(r.route_uid), str(raw / str(r.car_dir) / str(r.dongle_dir) / str(r.route_id))) for r in todo.itertuples(index=False)]
    t0, rows, n_err, n_done = time.time(), [], 0, 0
    pool = None
    if a.workers > 1:
        pool = Pool(a.workers, initializer=_init, initargs=(a.nice, a.max_segs), maxtasksperchild=200)
        results = pool.imap_unordered(_job, jobs, chunksize=4)
    else:
        _init(a.nice, a.max_segs)
        results = map(_job, jobs)
    try:
        for row in results:
            rows.append(row)
            n_done += 1
            n_err += bool(row["err"])
            if len(rows) >= a.batch:
                table = AP.merge_audit(table, AP.audit_frame(rows))
                AP.write_audit(table, out)
                rows = []
                print(f"[{n_done}/{len(jobs)}] errors so far {n_err}, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    if rows:
        table = AP.merge_audit(table, AP.audit_frame(rows))
        AP.write_audit(table, out)
    errs = table.loc[table["err"].astype(str) != "", "err"].astype(str).map(mask).value_counts().head(10).to_dict()
    print(f"done: probed {n_done} routes in {(time.time() - t0) / 60:.1f} min; table rows {len(table)}; errors this run {n_err}; "
          f"error types (table) {errs}", flush=True)
    return 2 if (n_done and n_err / n_done > ERR_MAX_FRAC) else 0


if __name__ == "__main__":
    sys.exit(main())
