#!/usr/bin/env python3
"""S5: build every public drive table, the drives index and the private QC frame.

reads   work/signals/<route_uid>/*.parquet + route_meta.json, build/drive_names.parquet, build/routes_labels.parquet,
        build/routes_all.csv (route list; falls back to drive_names when absent) or --routes <csv with a route_uid column>
writes  build/data/<vehicle_id>/<drive_id>.parquet (vtt.schema.DRIVE_COLUMNS), build/index/drives.parquet
        (vtt.schema.DRIVES_INDEX_COLUMNS, split columns null), private/qc/drives_private_qc.parquet
        (vtt.drives.PRIVATE_QC_COLUMNS), work/logs/s05_failures.csv (route_uid, status, masked error)

Idempotent / resumable: a drive whose parquet exists with the row count recorded in the index is skipped (--no-resume
rebuilds everything). The index and the QC frame are flushed every --flush-every finished drives and at the end, so an
interrupted run resumes from what it had. Prints counts only. Exit 2 when the acceptance check fails: more than
ACCEPT_MAX_FAIL_FRAC of the attempted routes failed, the index has schema problems or duplicate drive ids, or nothing was
built. Run it niced: nice -n 15 ionice -c3 $PY scripts/s05_build_drives.py --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402
from vtt import schema as S  # noqa: E402
from vtt.drives import PRIVATE_QC_COLUMNS, Roots, build_route, relabel_index  # noqa: E402
from vtt.private import ensure_private_root, write_parquet_atomic  # noqa: E402

ACCEPT_MAX_FAIL_FRAC = 0.005


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(C.DATA_ROOT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="build at most this many drives (after the resume skip)")
    ap.add_argument("--routes", default="", help="CSV with a route_uid column overriding build/routes_all.csv")
    ap.add_argument("--dry-run", action="store_true", help="plan only: print counts, write nothing")
    ap.add_argument("--no-resume", action="store_true", help="rebuild drives even when their parquet is complete")
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--relabel", action="store_true",
                    help="only refresh the label columns of build/index/drives.parquet from build/routes_labels.parquet "
                         "(no drive table is read or written; split columns are reset, run S6/S7b/S8 again)")
    return ap.parse_args(argv)


def relabel_main(roots: Roots) -> int:
    """--relabel: label-derived index columns from routes_labels (after S3c or any label change)."""
    for p in (roots.drives, roots.drive_names, roots.routes_labels):
        if not p.exists():
            print(f"missing {p.name}")
            return 2
    idx = pd.read_parquet(roots.drives)
    names = pd.read_parquet(roots.drive_names)
    labels = pd.read_parquet(roots.routes_labels)
    new, counts = relabel_index(idx, names, labels)
    S.write_public_parquet(new, roots.drives, S.DRIVES_INDEX_COLUMNS, "drives")
    problems = S.check_public_parquet(roots.drives, S.DRIVES_INDEX_COLUMNS, "drives")
    counts["size_known"] = int(new["size_class"].isin(C.SIZE_CLASSES).sum())
    counts["bev_ice"] = int(pd.Series(new["task_bev_ice_eligible"]).fillna(False).astype(bool).sum())
    print("relabel:", " ".join(f"{k}={v}" for k, v in counts.items()))
    if problems:
        print("index problems:", len(problems))
        return 2
    return 0


# ----------------------------------------------------------------------------- inputs
def load_names(roots: Roots) -> dict[str, dict]:
    """route_uid -> {vehicle_id, drive_id, group_id} from build/drive_names.parquet."""
    df = pd.read_parquet(roots.drive_names)
    need = {"route_uid", "vehicle_id", "drive_id", "group_id"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"drive_names.parquet lacks columns {sorted(missing)}")
    return {r["route_uid"]: {"vehicle_id": r["vehicle_id"], "drive_id": r["drive_id"], "group_id": r["group_id"]}
            for r in df[sorted(need)].to_dict("records")}


def load_labels(roots: Roots) -> dict[str, dict]:
    """route_uid -> label row (empty when build/routes_labels.parquet is absent)."""
    if not roots.routes_labels.exists():
        return {}
    df = pd.read_parquet(roots.routes_labels)
    return {r["route_uid"]: r for r in df.to_dict("records")}


def route_list(roots: Roots, routes_csv: str, names: dict[str, dict]) -> list[str]:
    """Routes to consider: --routes CSV, else build/routes_all.csv, else every named route; order preserved, de-duplicated."""
    if routes_csv:
        uids = pd.read_csv(routes_csv, dtype=str)["route_uid"].tolist()
    elif roots.routes_all.exists():
        uids = pd.read_csv(roots.routes_all, dtype=str)["route_uid"].tolist()
    else:
        uids = list(names)
    seen: set[str] = set()
    return [u for u in uids if isinstance(u, str) and u and not (u in seen or seen.add(u))]


def read_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:       # noqa: BLE001  a damaged partial file is rebuilt
        return None


def complete_drives(index: pd.DataFrame | None, roots: Roots) -> dict[str, int]:
    """drive_id -> n_rows of drives whose parquet exists with the row count recorded in the index."""
    out: dict[str, int] = {}
    if index is None or not len(index):
        return out
    for vid, did, n in zip(index["vehicle_id"], index["drive_id"], index["n_rows"]):
        p = roots.drive_file(vid, did)
        if p.exists():
            try:
                if pq.read_metadata(p).num_rows == int(n):
                    out[did] = int(n)
            except Exception:   # noqa: BLE001
                pass
    return out


def plan(routes, names, roots: Roots, done: dict[str, int], resume: bool) -> tuple[list[str], list[str], dict]:
    """(routes to build, drive_ids kept from the previous index, counts)."""
    todo, keep, counts = [], [], dict(listed=len(routes), no_name=0, no_signals=0, skipped_resume=0, to_build=0)
    for uid in routes:
        nm = names.get(uid)
        if nm is None:
            counts["no_name"] += 1
            continue
        if not (roots.signals / uid / "carState.parquet").exists():
            counts["no_signals"] += 1
            continue
        if resume and nm["drive_id"] in done:
            counts["skipped_resume"] += 1
            keep.append(nm["drive_id"])
            continue
        todo.append(uid)
    counts["to_build"] = len(todo)
    return todo, keep, counts


# ----------------------------------------------------------------------------- output
def assemble(roots: Roots, kept_index: pd.DataFrame | None, new_index: list[dict], kept_qc: pd.DataFrame | None,
             new_qc: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Write build/index/drives.parquet and private/qc/drives_private_qc.parquet from kept + new rows; returns (index, problems)."""
    parts = [p for p in (kept_index, pd.DataFrame(new_index) if new_index else None) if p is not None and len(p)]
    idx = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=S.names(S.DRIVES_INDEX_COLUMNS))
    idx = idx.reindex(columns=S.names(S.DRIVES_INDEX_COLUMNS))
    idx = idx.sort_values(["vehicle_id", "drive_id"], kind="stable").reset_index(drop=True)
    S.write_public_parquet(idx, roots.drives, S.DRIVES_INDEX_COLUMNS, "drives")
    problems = S.check_public_parquet(roots.drives, S.DRIVES_INDEX_COLUMNS, "drives")
    if idx["drive_id"].duplicated().any():
        problems.append(f"duplicate drive_id rows: {int(idx['drive_id'].duplicated().sum())}")
    qparts = [p for p in (kept_qc, pd.DataFrame(new_qc) if new_qc else None) if p is not None and len(p)]
    qc = pd.concat(qparts, ignore_index=True) if qparts else pd.DataFrame(columns=list(PRIVATE_QC_COLUMNS))
    qc = qc.reindex(columns=list(PRIVATE_QC_COLUMNS))
    write_parquet_atomic(qc, roots.private_qc, private=True)
    return idx, problems


def _work(job):
    return build_route(*job)


def main(argv=None) -> int:
    a = parse_args(argv)
    roots = Roots(a.data_root)
    if a.nice > 0:
        try:
            os.nice(a.nice)
        except OSError:
            pass
    if not roots.drive_names.exists():
        print(f"missing {roots.drive_names.relative_to(roots.data)} (run S4 first)")
        return 2
    if a.relabel:
        return relabel_main(roots)
    names = load_names(roots)
    labels = load_labels(roots)
    routes = route_list(roots, a.routes, names)
    kept_index = read_existing(roots.drives)
    kept_qc = read_existing(roots.private_qc)
    done = complete_drives(kept_index, roots) if not a.no_resume else {}
    todo, keep, counts = plan(routes, names, roots, done, resume=not a.no_resume)
    if a.limit > 0:
        todo = todo[: a.limit]
        counts["to_build"] = len(todo)
    counts["labels_available"] = sum(1 for u in todo if u in labels)
    print("plan:", " ".join(f"{k}={v}" for k, v in counts.items()))
    if a.dry_run:
        return 0

    ensure_private_root(roots.private)
    for d in (roots.data_dir, roots.index, roots.logs, roots.private_qc.parent):
        d.mkdir(parents=True, exist_ok=True)
    # rows of drives rebuilt in this run are replaced; every other existing row stays as long as its file still exists
    rebuilt = {names[u]["drive_id"] for u in todo}
    if kept_index is not None and len(kept_index):
        present = [roots.drive_file(v, d).exists() for v, d in zip(kept_index["vehicle_id"], kept_index["drive_id"])]
        kept_index = kept_index[~kept_index["drive_id"].isin(rebuilt) & np.asarray(present, bool)].reset_index(drop=True)
    if kept_qc is not None and len(kept_qc):
        kept_qc = kept_qc[kept_qc["drive_id"].isin(set(kept_index["drive_id"]) if kept_index is not None else set())].reset_index(drop=True)

    jobs = [(uid, names[uid], labels.get(uid), str(roots.data)) for uid in todo]
    new_index: list[dict] = []
    new_qc: list[dict] = []
    failures: list[dict] = []
    status = dict(ok=0, empty=0, no_signals=0, schema=0, error=0)
    t_start = time.time()
    problems: list[str] = []
    try:
        if jobs:
            with Pool(processes=max(1, a.workers)) as pool:
                for k, res in enumerate(pool.imap_unordered(_work, jobs, chunksize=1), 1):
                    status[res["status"]] = status.get(res["status"], 0) + 1
                    if res["status"] == "ok":
                        new_index.append(res["index"]); new_qc.append(res["qc"])
                    else:
                        failures.append(dict(route_uid=res["route_uid"], status=res["status"], err=res.get("err", "")))
                        if res["status"] == "schema" and res.get("index") is not None:
                            new_index.append(res["index"]); new_qc.append(res["qc"])
                    if a.flush_every > 0 and k % a.flush_every == 0:
                        assemble(roots, kept_index, new_index, kept_qc, new_qc)
                        print(f"progress: done={k}/{len(jobs)} ok={status['ok']} elapsed_s={time.time() - t_start:.0f}")
    finally:
        idx, problems = assemble(roots, kept_index, new_index, kept_qc, new_qc)
        if failures:
            pd.DataFrame(failures).to_csv(roots.failures, index=False)

    n_attempted = len(jobs)
    n_failed = status["error"] + status["schema"]
    fail_frac = n_failed / n_attempted if n_attempted else 0.0
    print("build:", " ".join(f"{k}={v}" for k, v in status.items()), f"index_rows={len(idx)}",
          f"elapsed_s={time.time() - t_start:.0f}")
    ok = True
    if problems:
        print("index problems:", len(problems)); ok = False
    if n_attempted and fail_frac > ACCEPT_MAX_FAIL_FRAC:
        print(f"failed fraction {fail_frac:.4f} > {ACCEPT_MAX_FAIL_FRAC}"); ok = False
    if len(idx) == 0:
        print("nothing built"); ok = False
    print("accept:", "ok" if ok else "FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
