#!/usr/bin/env python3
"""S1 -- census of the raw library, head probe of the new directories, logged-route resolution and the route lists.

Steps (plan §4 S1; helpers in vtt.manifest):
  1. private root: chmod 700, salt.txt and route_probe.parquet seeded from private/seeds/ when absent;
  2. census of EVERY route directory (s00.build_manifest, read-only) -> private/raw_manifest.csv;
  3. head probe of the directories missing from the probe cache (s01.probe_routes; stale rows -- 'no_logs' while the
     directory now holds a log, or 'error:*' -- are probed again unless --no-reprobe-stale) -> private/route_probe.parquet;
  4. dongle_log per route: probed initData id, else the extractor's static re-key table, else vtt.private.misfiled_dirs()
     extended by the ids learned this run from the folder's other routes (private/misfiled_dirs.json); a non-hex folder
     with no logged id anywhere is recorded there mapped to itself (needs a manual entry) and its routes are excluded;
  5. one row per (dongle_log, route_id) over ALL logged routes (s01.resolve_routes, no cohort filter); the same route_id
     under two different dongle_logs is compared by content: identical -> one copy kept (dup_rule identical_copy),
     different -> both kept and the pair appended to private/identity_merges.csv (dup_rule shared_route_id);
  6. outputs: private/uid_map.parquet (exactly tocf PRIVATE_UID_MAP), private/routes_resolved.parquet,
     build/routes_all.csv, build/routes_smoke20.csv, build/chunks/routes_chunk_NN.csv (+ routes_chunk_long.csv),
     build/logs/s01_summary.json (counts only).
Only counts are printed (identifiers masked). Exit code 2 when the acceptance checks fail (outputs are still written;
build/logs/s01_summary.json carries accepted=false).

  source env.sh && nice -n 15 ionice -c3 $PY scripts/s01_probe_uidmap.py [--data-root D] [--raw-root R] [--workers 8]
                                                                         [--limit N] [--no-reprobe-stale] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtt import config as C  # noqa: E402
from vtt import manifest as M  # noqa: E402
from vtt.private import ensure_private_root, mask, misfiled_dirs, write_json_atomic, write_parquet_atomic, write_text_atomic  # noqa: E402


def misfiled_json_update(private_root: Path, learned: dict, unresolved: list[str]) -> dict:
    """Current private/misfiled_dirs.json content merged with the ids learned this run (never overriding a manual 16-hex
    entry) and the unresolved folders mapped to themselves (the marker of a missing manual entry)."""
    p = private_root / "misfiled_dirs.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    out = dict(cur)
    for folder, did in learned.items():
        if not M.is_hex16(out.get(folder)):
            out[folder] = did
    for folder in unresolved:
        out.setdefault(folder, folder)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(C.DATA_ROOT), help="VTT data root (private/, build/ below it)")
    ap.add_argument("--raw-root", default=str(C.RAW_ROOT), help="read-only raw library")
    ap.add_argument("--workers", type=int, default=C.EXTRACT_WORKERS, help="probe processes")
    ap.add_argument("--limit", type=int, default=0, help="first N route directories only (tests)")
    ap.add_argument("--no-reprobe-stale", action="store_true", help="keep cached 'no_logs' / 'error:*' probe rows as they are")
    ap.add_argument("--dry-run", action="store_true", help="census + cached probe only; nothing is written")
    a = ap.parse_args(argv)
    data_root, raw_root = Path(a.data_root), Path(a.raw_root)
    priv, build = data_root / "private", data_root / "build"
    chunks_dir, logs_dir = build / "chunks", build / "logs"
    t0 = time.time()

    seeded = M.seed_private(priv)
    salt = seeded["salt"]
    print(f"private root ok; probe cache seeded now: {seeded['probe_seeded']}", flush=True)

    man, sha, n_files = M.census(raw_root, salt, a.limit)
    n_dirs = len(man)
    n_logged_dirs = int(((man.n_qlog + man.n_rlog) > 0).sum())
    print(f"census: {n_dirs} route dirs ({n_logged_dirs} with a log), {n_files} files, {int(man.n_segments.sum())} segments, "
          f"{man['bytes'].sum() / 1e9:.1f} GB, {time.time() - t0:.0f}s", flush=True)

    cache = priv / "route_probe.parquet"
    probe0 = M.load_probe(cache)
    stale = [] if a.no_reprobe_stale else M.stale_probe_paths(probe0, man)
    n_missing = len(set(man.path) - set(probe0.path)) + len(stale)
    if a.dry_run:
        probe, n_probed = probe0, 0
        print(f"probe: cache {len(probe0)} rows; {n_missing} would be probed ({len(stale)} stale); dry-run uses the cache only", flush=True)
    else:
        n_pruned = M.prune_probe_cache(cache, stale)
        probe, n_probed = M.probe_missing(man, raw_root, cache, salt, a.workers)
        print(f"probe: cache {len(probe0)} rows; probed now {n_probed} ({n_pruned} stale re-probed); cache {len(probe)} rows, "
              f"{time.time() - t0:.0f}s", flush=True)
    mp = M.attach_probe(man, probe)
    st = mp.probe_status[(mp.n_qlog + mp.n_rlog) > 0].value_counts().to_dict()
    n_ok = int(st.get("ok", 0))
    print(f"probe status of logged dirs: {st}; ok {100.0 * n_ok / max(1, n_logged_dirs):.1f}%", flush=True)

    learned, detail = M.folder_rekeys(mp)
    rekeys = dict(misfiled_dirs(priv))
    for k, v in learned.items():
        if not M.is_hex16(rekeys.get(k)):
            rekeys[k] = v
    mp = M.assign_dongle_log(mp, rekeys)
    ok_rows, nonhex_rows, junk = M.split_logged(mp)
    nonhex_folders_all = sorted({d for d in mp.dongle_dir if not M.is_hex16(d)})
    unresolved = sorted(set(nonhex_rows.dongle_dir))
    resolved_nonhex = sorted(set(d for d in ok_rows.dongle_dir if not M.is_hex16(d)))
    n_wrong_id = sum(1 for d in detail.values() if d["kind"] == "wrong_id" and d["status"] == "resolved")
    print(f"folders: {mp.dongle_dir.nunique()} dongle folders, {len(nonhex_folders_all)} non-hex (resolved {len(resolved_nonhex)}, "
          f"unresolved {len(unresolved)} -> {len(nonhex_rows)} logged routes excluded), {n_wrong_id} 16-hex folders re-keyed by their logs; "
          f"{len(junk)} log-less dirs left out", flush=True)

    rr = M.resolve_logged(ok_rows, salt)
    rr, merges = M.apply_duplicate_rules(rr, raw_root)
    rr["n_seg_rlog_only"] = M.rlog_only_counts(raw_root, rr)
    resolved = M.finish_resolved(rr)
    uid_map = M.make_uid_map(resolved)
    routes_all = M.routes_all_public(resolved)
    smoke = M.smoke_select(resolved)
    dup = resolved.dup_rule.value_counts().to_dict()
    n_multi = int((resolved.n_copies > 1).sum())
    print(f"routes: {len(uid_map)} logged routes (uid_map), n_copies>1 {n_multi}, dup rules {dup}, identity-merge pairs {len(merges)}, "
          f"long (> {C.LONG_ROUTE_SEGMENTS} segments) {int(resolved.is_long.sum())}, routes with rlog-only segments "
          f"{int((resolved.n_seg_rlog_only > 0).sum())}", flush=True)
    print(f"eras: {routes_all.software_era.value_counts().to_dict()}; devices: {routes_all.device_type.value_counts().to_dict()}", flush=True)

    if a.dry_run:
        print(f"dry-run: nothing written; smoke {len(smoke)} strata {smoke.stratum.value_counts().to_dict()}; "
              f"chunks would be {-(-int((~resolved.is_long).sum()) // C.CHUNK_ROUTES)} + long; {time.time() - t0:.0f}s", flush=True)
        return 0

    build.mkdir(parents=True, exist_ok=True)
    write_text_atomic(man.to_csv(index=False), priv / "raw_manifest.csv", private=True)
    write_parquet_atomic(uid_map, priv / "uid_map.parquet", private=True)
    write_parquet_atomic(resolved, priv / "routes_resolved.parquet", private=True)
    n_new_merges = M.append_identity_merges(priv / "identity_merges.csv", merges)
    write_json_atomic(misfiled_json_update(priv, learned, unresolved), priv / "misfiled_dirs.json", private=True)
    write_json_atomic(detail, priv / "misfiled_dirs_detail.json", private=True)
    routes_all.to_csv(build / "routes_all.csv", index=False)
    smoke.to_csv(build / "routes_smoke20.csv", index=False)
    ch = M.write_chunks(routes_all, chunks_dir)
    problems = M.acceptance(uid_map, resolved, routes_all, smoke, chunks_dir, [build / "routes_all.csv", build / "routes_smoke20.csv"])
    summary = {
        "stage": "S1", "config_version": C.CONFIG_VERSION, "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_listing_sha256": sha, "n_files": int(n_files), "n_dirs": int(n_dirs), "n_logged_dirs": n_logged_dirs,
        "n_logless_dirs": int(len(junk)), "n_probed_now": int(n_probed), "n_stale_reprobed": int(len(stale)),
        "probe_status_logged": {str(k): int(v) for k, v in st.items()}, "probe_ok_frac": n_ok / max(1, n_logged_dirs),
        "n_dongle_folders": int(mp.dongle_dir.nunique()), "n_nonhex_folders": len(nonhex_folders_all),
        "n_nonhex_resolved": len(resolved_nonhex), "n_nonhex_unresolved": len(unresolved),
        "n_routes_excluded_unresolved_folder": int(len(nonhex_rows)), "n_hex_folders_rekeyed": int(n_wrong_id),
        "n_routes": int(len(uid_map)), "n_multi_copy": n_multi, "dup_rules": {str(k): int(v) for k, v in dup.items()},
        "n_identity_merge_pairs": int(len(merges)), "n_identity_merges_new": int(n_new_merges),
        "n_long": int(resolved.is_long.sum()), "n_rlog_only_routes": int((resolved.n_seg_rlog_only > 0).sum()),
        "n_chunks": ch["n_chunks"], "chunk_files": ch["files"],
        "eras": {str(k): int(v) for k, v in routes_all.software_era.value_counts().items()},
        "devices": {str(k): int(v) for k, v in routes_all.device_type.value_counts().items()},
        "smoke": {str(k): int(v) for k, v in smoke.stratum.value_counts().items()},
        "problems": problems, "accepted": not problems, "elapsed_s": round(time.time() - t0, 1),
    }
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(summary, logs_dir / "s01_summary.json")
    ensure_private_root(priv)
    print(f"written: uid_map {len(uid_map)} rows, routes_all {len(routes_all)}, smoke {len(smoke)} strata {summary['smoke']}, "
          f"chunks {ch['n_chunks']} x <= {C.CHUNK_ROUTES} + long {ch['n_long']}, identity merges new {n_new_merges}; {time.time() - t0:.0f}s", flush=True)
    if problems:
        for p in problems:
            print("ERROR:", mask(p), flush=True)
        return 2
    if n_ok < 0.99 * n_logged_dirs:
        print(f"WARNING: probe ok below 99% of logged dirs ({n_ok}/{n_logged_dirs})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
