"""S1 helpers: software era, folder re-keys, logged-route resolution, duplicate rules, public route lists and chunks.

Everything that carries a real identifier (dongle id, route id, folder name, raw path) stays in frames that are written
only under the private root; the public frames (routes_all, routes_smoke20, chunks) hold route_uid + counts / categories
and `scan_public_text` asserts that no identifier-like token slipped through. Counts are plain integers (files, segments,
routes); nothing here has a time axis. The raw library is touched read-only: a directory census (s00.build_manifest),
head probes of directories missing from the probe cache (s01.probe_routes) and, for the few route ids that appear under
two different logged dongles, one decompressed log per copy (log_fingerprint).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt.tocf_bridge import load_script, tocf

_SCRIPTS: dict = {}


def script(name: str):
    """tocf scripts/<name>.py loaded once per process (a second exec would replace sys.modules[name] and break the
    pickling of its worker functions in multiprocessing pools)."""
    if name not in _SCRIPTS:
        _SCRIPTS[name] = load_script(name)
    return _SCRIPTS[name]

HEX16 = re.compile(r"^[0-9a-f]{16}$")
_VER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")          # fallback only when tocf.extract is not importable
DUP_RULES = ("none", "identical_copy", "shared_route_id")
PROBE_UNKNOWN = "unprobed"                                   # probe_status of a route that has no probe row (dry-run)
ROUTES_ALL_COLS = ["route_uid", "n_segments", "n_qlog", "n_rlog", "n_copies", "software_era", "device_type", "op_long",
                   "radar_unavailable", "probe_status", "is_long"]
RESOLVED_EXTRA = ["rel_path", "copies_rel", "n_copies", "n_segments", "n_qlog", "n_rlog", "version", "device_type", "car_fp",
                  "opLong", "radarUnavailable", "pcmCruise", "probe_status", "dup_rule", "is_long",
                  # beyond the PIPELINE.md list (documented in its Contract notes): useful to S3 / S2b, all private
                  "software_era", "route_style", "n_qcamera", "n_seg_rlog_only", "altExp", "expLongAvail", "git_commit", "dir_uid", "bytes"]
MERGE_COLS = ["dongle_log_a", "dongle_log_b", "route_id", "rule", "evidence", "route_uid_a", "route_uid_b", "n_seg_a", "n_seg_b",
              "added_by", "added_at"]
SMOKE_SPECIAL = ("nonhex_folder", "misfiled_dir", "date_style", "rlog_only", "multi_copy", "dup_rule", "long")
SMOKE_N = 20
SMOKE_PREF_SEGMENTS = (3, 40)                                # preferred route length (segments) of a smoke pick (extraction time)
CHUNK_GLOB = "routes_chunk_*.csv"


# ----------------------------------------------------------------------------- small pure helpers
def is_hex16(name) -> bool:
    """True when `name` is a 16-character lower-case hex string (the shape of a dongle id)."""
    return isinstance(name, str) and bool(HEX16.match(name))


def version_tuple(version) -> tuple[int, int, int] | None:
    """(major, minor, patch) of an openpilot version string via tocf.extract.version_tuple (regex fallback when tocf
    is not importable); None when no number is found."""
    try:
        return tocf("extract").version_tuple(version)
    except Exception:                      # pragma: no cover - tocf absent
        m = _VER_RE.search(str(version or ""))
        return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


def software_era(version) -> str:
    """Coarse writer generation of a version string: major 0 -> '0.8' / '0.9' / '0.10' / '0.11' by minor, calendar
    versions (major >= 2025) -> str(major), anything else (None, unparsable, 0.7, 1.x) -> 'other'."""
    vt = version_tuple(version)
    if vt is None:
        return "other"
    major, minor, _ = vt
    if major == 0:
        return {8: "0.8", 9: "0.9", 10: "0.10", 11: "0.11"}.get(minor, "other")
    if major >= 2025:
        return str(major)
    return "other"


def tri_state(series: pd.Series) -> np.ndarray:
    """Object array of True / False / None from a column that may hold bools, floats or NaN."""
    return np.array([None if (v is None or (isinstance(v, float) and np.isnan(v))) else bool(v) for v in series.to_numpy(dtype=object)],
                    dtype=object)


# ----------------------------------------------------------------------------- 1. private root seeding
def seed_private(private_root: Path) -> dict:
    """Create / tighten the private root, seed salt.txt (vtt.private.salt) and route_probe.parquet from seeds/ when
    absent. Returns {'salt': str, 'probe_seeded': bool}."""
    from vtt.private import ensure_private_root, salt as load_salt
    private_root = Path(private_root)
    ensure_private_root(private_root)
    s = load_salt(private_root)
    probe, seed = private_root / "route_probe.parquet", private_root / "seeds" / "route_probe.parquet"
    seeded = False
    if not probe.exists() and seed.exists():
        shutil.copyfile(seed, probe)
        os.chmod(probe, 0o600)
        seeded = True
    return {"salt": s, "probe_seeded": seeded}


# ----------------------------------------------------------------------------- 2. census + probe
def census(raw_root: Path, salt: str, limit: int = 0) -> tuple[pd.DataFrame, str, int]:
    """Every route directory of the flat raw layout with its file counts (s00.build_manifest, read-only scandir).
    Returns (manifest with s00.PRIVATE_COLS, sha256 of the sorted 'relpath,size' listing, number of files)."""
    s00 = script("s00_raw_manifest")
    man, sha, n_files = s00.build_manifest(Path(raw_root), salt, limit)
    for c in ("path", "car_dir", "dongle_dir", "route_id"):
        man[c] = man[c].astype(str)
    return man, sha, n_files


def load_probe(cache: Path) -> pd.DataFrame:
    """The probe cache (s01.PROBE_COLS) or an empty frame with those columns."""
    s01 = script("s01_freeze_inputs")
    return pd.read_parquet(cache) if Path(cache).exists() else pd.DataFrame(columns=s01.PROBE_COLS)


def stale_probe_paths(probe: pd.DataFrame, man: pd.DataFrame) -> list[str]:
    """Cached probe rows that must be probed again: status 'no_logs' while the manifest now holds a log in that
    directory (files added since the seed was taken), or an 'error:*' status (a transient read problem)."""
    if not len(probe):
        return []
    logged = set(man.path[(man.n_qlog + man.n_rlog) > 0])
    st = probe.probe_status.fillna("").astype(str)
    stale = ((st == "no_logs") & probe.path.isin(logged)) | st.str.startswith("error")
    return probe.path[stale].tolist()


def prune_probe_cache(cache: Path, paths: list[str]) -> int:
    """Rewrite the cache without `paths` (atomic, mode 600) so s01.probe_routes probes them again. Returns the count."""
    if not paths or not Path(cache).exists():
        return 0
    from vtt.private import write_parquet_atomic
    probe = pd.read_parquet(cache)
    keep = probe[~probe.path.isin(set(paths))]
    write_parquet_atomic(keep, Path(cache), private=True)
    return int(len(probe) - len(keep))


def probe_missing(man: pd.DataFrame, raw_root: Path, cache: Path, salt: str, workers: int) -> tuple[pd.DataFrame, int]:
    """Head-probe the manifest directories absent from the cache (s01.probe_routes: initData + carParams of the first
    readable log, VIN only as a salted hash) and return (full probe frame, number probed now)."""
    s01 = script("s01_freeze_inputs")
    old = load_probe(cache)
    n_new = len(set(man.path) - set(old.path))
    probe = s01.probe_routes(man, Path(raw_root), Path(cache), salt, max(1, int(workers)))
    if Path(cache).exists():
        try:
            os.chmod(cache, 0o600)
        except OSError:
            pass
    return probe, n_new


def attach_probe(man: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    """Manifest + probe columns (left join on path; probe_seg dropped; missing rows get probe_status PROBE_UNKNOWN).
    The probe's dongle_log column is renamed dongle_probe: dongle_log is assigned later (assign_dongle_log)."""
    cols = [c for c in probe.columns if c not in ("probe_seg",)]
    out = man.merge(probe[cols].drop_duplicates("path", keep="last"), on="path", how="left")
    out = out.rename(columns={"dongle_log": "dongle_probe"})
    out["probe_status"] = out.probe_status.where(out.probe_status.notna(), PROBE_UNKNOWN).astype(str)
    return out


# ----------------------------------------------------------------------------- 3. folder re-keys and dongle_log
def folder_rekeys(mp: pd.DataFrame) -> tuple[dict[str, str], dict[str, dict]]:
    """Learn folder -> logged dongle id from the probed routes of every dongle folder (mp = attach_probe output).
    A non-hex folder (a person's name) resolves to the id its logged routes carry (the most frequent one when several;
    status 'ambiguous'); a 16-hex folder whose logged routes all name ONE other id is a misfiled id (kind 'wrong_id');
    a non-hex folder with no logged route at all is 'unresolved' (needs a manual entry). Returns (rekeys, detail) with
    detail[folder] = {kind, status, n_routes, n_logged, n_ids, ids: {id: n}} (real ids: private use only)."""
    rekeys, detail = {}, {}
    for folder, g in mp.groupby("dongle_dir", sort=True):
        ids = g.dongle_probe[g.dongle_probe.map(is_hex16)].value_counts()
        hex_folder = is_hex16(folder)
        kind = "own_id" if hex_folder else "name"
        status = "own"
        if not hex_folder:
            if len(ids) == 0:
                status = "unresolved"
            else:
                status = "resolved" if len(ids) == 1 else "ambiguous"
                rekeys[folder] = str(ids.index[0])
        elif len(ids) and folder not in ids.index:
            kind = "wrong_id"
            if len(ids) == 1:
                status = "resolved"
                rekeys[folder] = str(ids.index[0])
            else:
                status = "ambiguous"
        if status != "own":
            detail[folder] = {"kind": kind, "status": status, "n_routes": int(len(g)),
                              "n_logged": int(((g.n_qlog + g.n_rlog) > 0).sum()), "n_ids": int(len(ids)),
                              "ids": {str(k): int(v) for k, v in ids.items()}}
    return rekeys, detail


def resolve_dongle_log(dongle_dir: str, probed, rekeys: dict[str, str]) -> str:
    """Logged dongle id of one route: the probed initData id when it is a valid id, else rawlib.dongle_from_dir
    (the extractor's static re-key table, else the folder), else the extended re-key table (vtt.private.misfiled_dirs
    + the ids learned this run). A folder mapped to itself (unresolved) stays a non-hex name."""
    rawlib = tocf("rawlib")
    logged = probed if is_hex16(probed) else None
    dl = rawlib.dongle_from_dir(dongle_dir, logged)
    if dl == dongle_dir and dongle_dir in rekeys:
        dl = rekeys[dongle_dir]
    return str(dl)


def assign_dongle_log(mp: pd.DataFrame, rekeys: dict[str, str]) -> pd.DataFrame:
    """mp + dongle_log column (resolve_dongle_log per row)."""
    out = mp.copy()
    out["dongle_log"] = [resolve_dongle_log(d, p, rekeys) for d, p in zip(out.dongle_dir, out.dongle_probe)]
    return out


def split_logged(mp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(resolvable logged routes, logged routes whose dongle_log is still a non-hex name, log-less dirs)."""
    logged = (mp.n_qlog + mp.n_rlog) > 0
    ok = mp.dongle_log.map(is_hex16)
    return mp[logged & ok].copy(), mp[logged & ~ok].copy(), mp[~logged].copy()


# ----------------------------------------------------------------------------- 4. resolution of copies + duplicate rules
def resolve_logged(man_logged: pd.DataFrame, salt: str) -> pd.DataFrame:
    """One row per (dongle_log, route_id): the chosen copy (s01.resolve_routes -> rawlib.choose_candidate: fingerprint
    match, then qlog + qcamera count, rlog count, lexical) with n_copies, route_uid, identity_uid, rel_path and
    copies_rel (JSON list of every copy's car/dongle/route path)."""
    s01 = script("s01_freeze_inputs")
    if not len(man_logged):
        return pd.DataFrame(columns=list(man_logged.columns) + ["n_copies", "route_uid", "identity_uid", "rel_path", "copies_rel"])
    rr = s01.resolve_routes(man_logged, salt)
    copies = (man_logged.groupby(["dongle_log", "route_id"]).path.apply(lambda s: json.dumps(sorted(s)))
              .rename("copies_rel").reset_index())
    rr = rr.merge(copies, on=["dongle_log", "route_id"], how="left")
    rr["rel_path"] = rr["path"].astype(str)
    return rr


def log_fingerprint(raw_root: Path, rel_path: str) -> tuple[tuple[int, ...], str | None]:
    """Content fingerprint of one raw route dir: (sorted segment numbers, 'kind:seg:sha256' of the decompressed
    lowest-numbered qlog -- rlog when that segment has no qlog -- or None when no log is readable)."""
    rawlib = tocf("rawlib")
    segs = rawlib.segment_table(Path(raw_root) / rel_path)
    nums = tuple(sorted(int(n) for n in segs))
    for n in nums:
        f = segs[n]
        log = f["qlog"] or f["rlog"]
        if log is None:
            continue
        kind = "qlog" if f["qlog"] else "rlog"
        try:
            data = rawlib.read_bytes(log)
        except OSError:
            return nums, None
        return nums, f"{kind}:{n}:{hashlib.sha256(data).hexdigest()}"
    return nums, None


def _copy_preference(row) -> tuple:
    """Sort key (higher = preferred) of a copy among identical copies: folder equals its logged id, probe ok, a 16-hex
    folder, more qlog + qcamera files, more rlogs, then the lexically smaller path."""
    return (int(row.dongle_dir == row.dongle_log), int(str(row.probe_status) == "ok"), int(is_hex16(row.dongle_dir)),
            int(row.n_qlog) + int(row.n_qcamera), int(row.n_rlog), tuple(-ord(ch) for ch in str(row.rel_path)))


def apply_duplicate_rules(rr: pd.DataFrame, raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same route_id under two (or more) different dongle_logs. Copies with identical content (same segment-number set
    and same lowest-numbered log, log_fingerprint) are one recording: the preferred copy (_copy_preference, i.e. the
    folder that equals the logged id) stays with dup_rule 'identical_copy' and absorbs the other's copies; different
    content keeps both rows with dup_rule 'shared_route_id' and yields one identity-merge pair (MERGE_COLS; evidence
    'counter_id' | 'date_id_same_car' | 'date_id_other_car' so a consumer may weigh date collisions). Returns
    (rr with dup_rule and the dropped copies removed, merges)."""
    rr = rr.copy()
    rr["dup_rule"] = "none"
    merges, dropped = [], set()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for rid, g in rr.groupby("route_id", sort=True):
        if g.dongle_log.nunique() < 2:
            continue
        fps = {i: log_fingerprint(raw_root, rr.at[i, "rel_path"]) for i in g.index}
        for a, b in combinations(sorted(g.index), 2):
            if a in dropped or b in dropped:
                continue
            fa, fb = fps[a], fps[b]
            if fa[1] is not None and fa == fb:
                keep, lose = (a, b) if _copy_preference(rr.loc[a]) >= _copy_preference(rr.loc[b]) else (b, a)
                rr.at[keep, "dup_rule"] = "identical_copy"
                rr.at[keep, "n_copies"] = int(rr.at[keep, "n_copies"]) + int(rr.at[lose, "n_copies"])
                rr.at[keep, "copies_rel"] = json.dumps(sorted(set(json.loads(rr.at[keep, "copies_rel"]) + json.loads(rr.at[lose, "copies_rel"]))))
                dropped.add(lose)
            else:
                ra, rb = rr.loc[a], rr.loc[b]
                rr.at[a, "dup_rule"] = rr.at[b, "dup_rule"] = "shared_route_id"
                style = tocf("rawlib").route_style(str(rid))
                evidence = "counter_id" if style == "counter" else ("date_id_same_car" if ra.car_dir == rb.car_dir else "date_id_other_car")
                x, y = sorted([(str(ra.dongle_log), str(ra.route_uid), int(ra.n_segments)), (str(rb.dongle_log), str(rb.route_uid), int(rb.n_segments))])
                merges.append({"dongle_log_a": x[0], "dongle_log_b": y[0], "route_id": str(rid), "rule": "shared_route_id",
                               "evidence": evidence, "route_uid_a": x[1], "route_uid_b": y[1], "n_seg_a": x[2], "n_seg_b": y[2],
                               "added_by": "s01_probe_uidmap", "added_at": now})
    out = rr.drop(index=sorted(dropped)).reset_index(drop=True)
    return out, pd.DataFrame(merges, columns=MERGE_COLS)


def append_identity_merges(path: Path, merges: pd.DataFrame) -> int:
    """Append-only union into private/identity_merges.csv keyed by (dongle_log_a, dongle_log_b, route_id). Returns the
    number of new rows."""
    from vtt.private import write_text_atomic
    path = Path(path)
    old = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=MERGE_COLS)
    for c in MERGE_COLS:
        if c not in old.columns:
            old[c] = None
    key = ["dongle_log_a", "dongle_log_b", "route_id"]
    have = set(map(tuple, old[key].astype(str).to_numpy().tolist())) if len(old) else set()
    new = merges[[tuple(map(str, r)) not in have for r in merges[key].to_numpy().tolist()]] if len(merges) else merges
    if len(new) == 0 and path.exists():
        return 0
    out = pd.concat([old[MERGE_COLS], new[MERGE_COLS].astype(object)], ignore_index=True) if len(old) else new[MERGE_COLS]
    write_text_atomic(out.to_csv(index=False), path, private=True)
    return int(len(new))


def rlog_only_counts(raw_root: Path, rr: pd.DataFrame) -> pd.Series:
    """Exact number of segments that hold an rlog but no qlog (the extractor's rlog_dec path), from one directory
    listing (rawlib.segment_table) of every route whose counts allow such a segment (n_rlog > 0 and
    n_qlog < n_segments); 0 elsewhere. Index aligned with rr."""
    rawlib = tocf("rawlib")
    out = pd.Series(0, index=rr.index, dtype="int64")
    n_rlog, n_qlog, n_seg = (pd.to_numeric(rr[c], errors="coerce").fillna(0).astype(int) for c in ("n_rlog", "n_qlog", "n_segments"))
    for i in rr.index[(n_rlog > 0) & (n_qlog < n_seg)]:
        try:
            segs = rawlib.segment_table(Path(raw_root) / str(rr.at[i, "rel_path"]))
        except OSError:
            continue
        out.at[i] = sum(1 for f in segs.values() if f["rlog"] is not None and f["qlog"] is None)
    return out


# ----------------------------------------------------------------------------- 5. tables
def finish_resolved(rr: pd.DataFrame) -> pd.DataFrame:
    """routes_resolved: tocf PRIVATE_UID_MAP columns + RESOLVED_EXTRA (software_era from version, is_long from
    config.LONG_ROUTE_SEGMENTS), sorted by route_uid."""
    S = tocf("schema")
    rr = rr.copy()
    rr["software_era"] = rr.version.map(software_era)
    rr["is_long"] = rr.n_segments.astype(int) > int(C.LONG_ROUTE_SEGMENTS)
    if "n_seg_rlog_only" not in rr:
        rr["n_seg_rlog_only"] = 0
    for c in ("n_copies", "n_segments", "n_qlog", "n_rlog", "n_qcamera", "bytes", "n_seg_rlog_only"):
        rr[c] = pd.to_numeric(rr[c], errors="coerce").fillna(0).astype("int64")
    if "altExp" in rr:
        rr["altExp"] = pd.to_numeric(rr.altExp, errors="coerce")
    cols = list(S.PRIVATE_UID_MAP) + RESOLVED_EXTRA
    for c in cols:
        if c not in rr.columns:
            rr[c] = None
    out = rr[cols].sort_values("route_uid").reset_index(drop=True)
    for c in list(S.PRIVATE_UID_MAP) + ["version", "device_type", "car_fp", "git_commit", "probe_status", "rel_path", "copies_rel"]:
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out


def make_uid_map(resolved: pd.DataFrame) -> pd.DataFrame:
    """EXACTLY tocf.schema.PRIVATE_UID_MAP columns (object dtype, None for missing), one row per kept logged route."""
    S = tocf("schema")
    cols = list(S.PRIVATE_UID_MAP)
    um = resolved[cols].copy()
    for c in cols:
        um[c] = um[c].astype(object).where(um[c].notna(), None)
    return um.reset_index(drop=True)


def routes_all_public(resolved: pd.DataFrame) -> pd.DataFrame:
    """build/routes_all.csv rows (ROUTES_ALL_COLS): internal route_uid + counts / categories, nothing else. op_long and
    radar_unavailable are tri-state (True / False / empty when the probe did not read carParams); device_type 'unknown'
    when absent."""
    return pd.DataFrame({
        "route_uid": resolved.route_uid.astype(str),
        "n_segments": resolved.n_segments.astype(int), "n_qlog": resolved.n_qlog.astype(int), "n_rlog": resolved.n_rlog.astype(int),
        "n_copies": resolved.n_copies.astype(int),
        "software_era": resolved.software_era.astype(str),
        "device_type": [d if isinstance(d, str) and d else "unknown" for d in resolved.device_type],
        "op_long": tri_state(resolved.opLong), "radar_unavailable": tri_state(resolved.radarUnavailable),
        "probe_status": resolved.probe_status.astype(str), "is_long": resolved.is_long.astype(bool),
    }, columns=ROUTES_ALL_COLS)


def _prefer(cand: pd.DataFrame) -> pd.DataFrame:
    """Deterministic order of smoke candidates: probe ok first, preferred length (SMOKE_PREF_SEGMENTS) first, then route_uid."""
    lo, hi = SMOKE_PREF_SEGMENTS
    k_ok = (cand.probe_status.astype(str) != "ok").astype(int)
    k_len = (~cand.n_segments.between(lo, hi)).astype(int)
    return cand.assign(_k0=k_ok, _k1=k_len).sort_values(["_k0", "_k1", "route_uid"]).drop(columns=["_k0", "_k1"])


def smoke_select(resolved: pd.DataFrame, n: int = SMOKE_N) -> pd.DataFrame:
    """<= n routes for the real-data smoke run: one per special stratum (SMOKE_SPECIAL: route under a non-hex folder,
    re-keyed folder, date-style id, a segment with rlog only (n_seg_rlog_only), n_copies > 1, a duplicate rule, a long route), then one
    per (software_era x device_type) cell round-robin (largest cells first) until n. Returns routes_all columns +
    'stratum'; deterministic (sorted by route_uid, no randomness)."""
    pub = routes_all_public(resolved).set_index("route_uid")
    r = resolved.set_index("route_uid")
    long_thr = int(C.LONG_ROUTE_SEGMENTS)
    masks = {
        "nonhex_folder": ~r.dongle_dir.map(is_hex16),
        "misfiled_dir": r.dongle_dir != r.dongle_log,
        "date_style": r.route_style == "date",
        "rlog_only": r.n_seg_rlog_only > 0,
        "multi_copy": r.n_copies > 1,
        "dup_rule": r.dup_rule != "none",
        "long": r.n_segments > long_thr,
    }
    taken, picks = [], []
    for name in SMOKE_SPECIAL:
        cand = r[masks[name] & ~r.index.isin(taken)]
        if not len(cand):
            continue
        if name == "long":                                   # the shortest long route keeps the smoke run short
            uid = cand.reset_index().sort_values(["n_segments", "route_uid"]).route_uid.iloc[0]
        elif name == "misfiled_dir":                         # prefer a re-keyed 16-hex folder (the non-hex one has its own stratum)
            cand = _prefer(cand.reset_index())
            cand = cand.assign(_h=(~cand.dongle_dir.map(is_hex16)).astype(int)).sort_values("_h", kind="stable")
            uid = cand.route_uid.iloc[0]
        else:
            uid = _prefer(cand.reset_index()).route_uid.iloc[0]
        taken.append(uid)
        picks.append((uid, name))
        if len(picks) >= n:
            break
    rest = r[~r.index.isin(taken)].reset_index()
    rest = rest.assign(software_era=rest.version.map(software_era), _dev=[d if isinstance(d, str) and d else "unknown" for d in rest.device_type])
    cells = [(k, _prefer(g)) for k, g in rest.groupby(["software_era", "_dev"], sort=True)]
    cells.sort(key=lambda kg: (-len(kg[1]), kg[0]))
    pos = {k: 0 for k, _ in cells}
    while len(picks) < n and cells:
        progressed = False
        for k, g in cells:
            if len(picks) >= n:
                break
            if pos[k] < len(g):
                uid = g.route_uid.iloc[pos[k]]
                pos[k] += 1
                picks.append((uid, f"era_{k[0]}_{k[1]}"))
                progressed = True
        if not progressed:
            break
    out = pub.loc[[u for u, _ in picks]].reset_index()
    out["stratum"] = [s for _, s in picks]
    return out[ROUTES_ALL_COLS + ["stratum"]]


def write_chunks(routes_all: pd.DataFrame, chunks_dir: Path, chunk_routes: int = C.CHUNK_ROUTES) -> dict:
    """build/chunks/routes_chunk_NN.csv (route_uid column, <= chunk_routes rows, sorted by route_uid) for the routes with
    is_long False and routes_chunk_long.csv (always written, possibly header-only) for the long ones. Stale chunk files
    are removed first. Returns {'n_chunks', 'n_long', 'files'}."""
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old in chunks_dir.glob(CHUNK_GLOB):
        old.unlink()
    short = sorted(routes_all.route_uid[~routes_all.is_long.astype(bool)].astype(str))
    long_ = sorted(routes_all.route_uid[routes_all.is_long.astype(bool)].astype(str))
    files = []
    for i in range(0, len(short), int(chunk_routes)):
        p = chunks_dir / f"routes_chunk_{i // int(chunk_routes):02d}.csv"
        pd.DataFrame({"route_uid": short[i:i + int(chunk_routes)]}).to_csv(p, index=False)
        files.append(p.name)
    p = chunks_dir / "routes_chunk_long.csv"
    pd.DataFrame({"route_uid": long_}).to_csv(p, index=False)
    files.append(p.name)
    return {"n_chunks": len(files) - 1, "n_long": len(long_), "files": files}


# ----------------------------------------------------------------------------- 6. acceptance
def scan_public_text(paths: list[Path]) -> dict[str, int]:
    """{file name: number of identifier-like tokens} (vtt.private HEX16 / ROUTE_DATE / ROUTE_NUM) over text files."""
    from vtt.private import HEX16 as H16, ROUTE_DATE, ROUTE_NUM
    out = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        n = len(H16.findall(text)) + len(ROUTE_DATE.findall(text)) + len(ROUTE_NUM.findall(text))
        if n:
            out[p.name] = n
    return out


def acceptance(uid_map: pd.DataFrame, resolved: pd.DataFrame, routes_all: pd.DataFrame, smoke: pd.DataFrame,
               chunks_dir: Path, public_csvs: list[Path]) -> list[str]:
    """Structural checks of the S1 outputs; returns problem strings (empty = accepted). Uses only counts in messages."""
    S = tocf("schema")
    problems = []
    if list(uid_map.columns) != list(S.PRIVATE_UID_MAP):
        problems.append("uid_map columns differ from tocf PRIVATE_UID_MAP")
    if uid_map.route_uid.duplicated().any():
        problems.append(f"{int(uid_map.route_uid.duplicated().sum())} duplicated route_uid in uid_map")
    if not uid_map.dongle_log.map(is_hex16).all():
        problems.append(f"{int((~uid_map.dongle_log.map(is_hex16)).sum())} uid_map rows with a non-hex dongle_log")
    if uid_map.route_id.isna().any() or (uid_map.route_id.astype(str) == "").any():
        problems.append("uid_map rows without route_id")
    if len(resolved) != len(uid_map) or set(resolved.route_uid) != set(uid_map.route_uid):
        problems.append("routes_resolved and uid_map disagree on the route set")
    if resolved.rel_path.duplicated().any():
        problems.append(f"{int(resolved.rel_path.duplicated().sum())} duplicated rel_path in routes_resolved")
    if not resolved.dup_rule.isin(DUP_RULES).all():
        problems.append("unknown dup_rule value")
    if list(routes_all.columns) != ROUTES_ALL_COLS or set(routes_all.route_uid) != set(uid_map.route_uid):
        problems.append("routes_all columns or route set differ from uid_map")
    if not set(smoke.route_uid) <= set(routes_all.route_uid) or len(smoke) > SMOKE_N or smoke.route_uid.duplicated().any():
        problems.append("smoke list is not a <= 20-row subset of routes_all")
    files = sorted(Path(chunks_dir).glob(CHUNK_GLOB))
    seen: list[str] = []
    for f in files:
        df = pd.read_csv(f, dtype=str)
        if list(df.columns) != ["route_uid"]:
            problems.append(f"chunk {f.name} columns != [route_uid]")
        if f.name != "routes_chunk_long.csv" and len(df) > int(C.CHUNK_ROUTES):
            problems.append(f"chunk {f.name} holds more than CHUNK_ROUTES rows")
        seen += df.route_uid.dropna().tolist()
    if len(seen) != len(set(seen)) or set(seen) != set(uid_map.route_uid.astype(str)):
        problems.append("chunk files are not a disjoint cover of uid_map")
    if not files or files[-1].name != "routes_chunk_long.csv":
        problems.append("routes_chunk_long.csv missing")
    hits = scan_public_text(list(public_csvs) + files)
    if hits:
        problems.append(f"identifier-like tokens in public csv: {sum(hits.values())} in {len(hits)} files")
    return problems
