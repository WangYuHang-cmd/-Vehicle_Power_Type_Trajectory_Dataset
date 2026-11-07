"""S2c: rebuild work/tables/segments.parquet (tocf.schema.SEGMENTS) from the extractor's per-route output.

Why a rebuild: the extractor appends to segments.parquet per run and per chunk; after the hardlink reuse of an earlier
extraction plus several chunked runs the table must be derived once from what is actually on disk. The segment rows come
from route_meta.json (tocf.tables.build_segments), the per-topic rates are recomputed from the parquets with the extractor's
own formula (tocf.decim.measured_rate), and two side tables are written for the acceptance step: per-segment row counts and
a per-route meta index. Units: rates in Hz, spans in s, logMonoTime in ns (boot clock, never published).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt.tocf_bridge import tocf

SEG_RATE_TOPICS = {"rate_carState": "carState", "rate_radarState": "radarState", "rate_dm": "driverMonitoringState"}
COUNT_TOPICS = ("carState", "radarState", "pose", "controlsState", "carControl", "selfdriveState", "driverMonitoringState")
SEGMENT_COUNTS_COLUMNS = ["route_uid", "seg", "src_kind", "span_s"] + [f"n_{t}" for t in COUNT_TOPICS]
ROUTES_META_COLUMNS = ["route_uid", "version", "software_era", "device_type", "extractor_version", "config_version", "n_seg",
                       "n_seg_qlog", "n_seg_rlog_dec", "n_seg_none", "n_truncated", "n_read_error", "pose_source", "car_fp",
                       "opLong", "radarUnavailable", "pcmCruise", "altExp"]
SEGMENT_COUNTS_FILE = "segment_counts.parquet"
ROUTES_META_FILE = "routes_meta.parquet"


# ----------------------------------------------------------------------------- software era
_OWNER: list = []          # [callable] once resolved; [None] when the S1 module has no software_era


def software_era(version) -> str:
    """config.SOFTWARE_ERAS bucket of an openpilot version string: '0.8' .. '0.11' by major.minor, '2025' / '2026' for the
    calendar-style writers, else 'other' (None / unparsable included). Delegates to vtt.manifest.software_era when the S1
    module provides one (single owner of the public routes_all.csv column); the lookup happens once per process. Whatever the
    owner answers is normalised to config.SOFTWARE_ERAS (a calendar writer the tuple does not list, e.g. 2027, is 'other'),
    so the acceptance never meets an era outside the tuple."""
    if not _OWNER:
        try:
            from vtt.manifest import software_era as _owner   # noqa: PLC0415  (S1 module, written in parallel)
            _OWNER.append(_owner)
        except ImportError:
            _OWNER.append(None)
    era = _OWNER[0](version) if _OWNER[0] is not None else software_era_fallback(version)
    return era if era in C.SOFTWARE_ERAS else "other"


def software_era_fallback(version) -> str:
    vt = tocf("extract").version_tuple(version)
    if vt is None:
        return "other"
    major, minor, _ = vt
    era = f"{major}.{minor}" if major == 0 else str(major)
    return era if era in C.SOFTWARE_ERAS else "other"


# ----------------------------------------------------------------------------- per-route readers
def list_extracted(signals_dir: Path) -> list[str]:
    """Sorted route_uids with a route_meta.json under signals_dir (the '.<uid>.building' dirs of a crashed run are skipped)."""
    signals_dir = Path(signals_dir)
    if not signals_dir.exists():
        return []
    out = []
    for d in signals_dir.iterdir():
        if d.name.startswith("."):
            continue
        try:
            if (d / "route_meta.json").exists():
                out.append(d.name)
        except OSError:                             # an unreadable directory is not an extracted route
            continue
    return sorted(out)


def read_meta(signals_dir: Path, uid: str) -> dict:
    return json.loads((Path(signals_dir) / uid / "route_meta.json").read_text())


def _topic_groups(signals_dir: Path, uid: str, topic: str) -> dict[int, np.ndarray]:
    """{seg: sorted logMonoTime array} of one topic parquet (empty when the file or its rows are missing)."""
    p = Path(signals_dir) / uid / f"{topic}.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p, columns=["logMonoTime", "seg"])
    if not len(df):
        return {}
    return {int(s): np.sort(g.to_numpy(np.int64)) for s, g in df.groupby("seg")["logMonoTime"]}


def _rate(times: np.ndarray) -> float:
    """tocf.decim.measured_rate on a time array: (n - 1) / span_s, NaN below 2 rows or with a zero span."""
    return tocf("decim").measured_rate(pd.DataFrame({"logMonoTime": times}))


def route_tables(signals_dir: Path, uid: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(segments rows typed as tocf.schema.SEGMENTS with recomputed rates, segment_counts rows, routes_meta row) of one route."""
    signals_dir = Path(signals_dir)
    tables = tocf("tables")
    seg = tables.build_segments(signals_dir.parent, [uid])          # tocf layout: <data_root>/signals/<uid>
    meta = read_meta(signals_dir, uid)
    groups = {t: _topic_groups(signals_dir, uid, t) for t in COUNT_TOPICS}
    for col, topic in SEG_RATE_TOPICS.items():
        seg[col] = [_rate(groups[topic][s]) if s in groups[topic] else np.nan for s in seg["seg"].to_numpy()]
    span = (seg["t1_mono_ns"].to_numpy(np.int64) - seg["t0_mono_ns"].to_numpy(np.int64)) / 1e9
    span = np.where((seg["t0_mono_ns"].to_numpy() >= 0) & (seg["t1_mono_ns"].to_numpy() >= 0), span, np.nan)
    counts = pd.DataFrame({"route_uid": uid, "seg": seg["seg"].to_numpy(np.int64), "src_kind": seg["src_kind"].to_numpy(),
                           "span_s": span.astype(np.float64)})
    for t in COUNT_TOPICS:
        counts[f"n_{t}"] = np.asarray([len(groups[t].get(int(s), ())) for s in seg["seg"]], dtype=np.int64)
    return seg, counts, meta_row(uid, meta)


def meta_row(uid: str, meta: dict) -> dict:
    """ROUTES_META_COLUMNS row of a route_meta.json (segment flags summarised; no identifier-bearing field is copied)."""
    segs = [s for s in (meta.get("segments") or []) if isinstance(s, dict)]
    kinds = [str(s.get("src_kind", "")) for s in segs]
    logs = [str((s.get("flags") or {}).get("log", "")) for s in segs]
    return {"route_uid": uid, "version": meta.get("version"), "software_era": software_era(meta.get("version")),
            "device_type": meta.get("device_type"), "extractor_version": meta.get("extractor_version"),
            "config_version": meta.get("config_version"), "n_seg": int(meta.get("n_seg") or len(segs)),
            "n_seg_qlog": kinds.count("qlog"), "n_seg_rlog_dec": kinds.count("rlog_dec"), "n_seg_none": kinds.count("none"),
            "n_truncated": sum("truncated" in f for f in logs), "n_read_error": sum("read_error" in f for f in logs),
            "pose_source": meta.get("pose_source"), "car_fp": meta.get("car_fp"), "opLong": meta.get("opLong"),
            "radarUnavailable": meta.get("radarUnavailable"), "pcmCruise": meta.get("pcmCruise"), "altExp": meta.get("altExp")}


# ----------------------------------------------------------------------------- assembly
def _job(args) -> tuple:
    signals_dir, uid = args
    try:
        seg, counts, row = route_tables(signals_dir, uid)
        return uid, seg, counts, row, ""
    except Exception as e:                      # noqa: BLE001  one unreadable route must not stop the rebuild
        return uid, None, None, None, type(e).__name__


def rebuild(signals_dir: Path, uids: list[str] | None = None, workers: int = 1) -> dict:
    """Rebuild over uids (default: every extracted route). Returns {'segments', 'counts', 'routes_meta', 'errors'}:
    the three frames (segments typed as tocf.schema.SEGMENTS) and {uid: exception type} of routes that could not be read."""
    signals_dir = Path(signals_dir)
    uids = list_extracted(signals_dir) if uids is None else list(uids)
    jobs = [(signals_dir, u) for u in uids]
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=_nice) as pool:
            results = list(pool.imap_unordered(_job, jobs, chunksize=8))
    else:
        results = [_job(j) for j in jobs]
    results.sort(key=lambda r: r[0])
    segs = [r[1] for r in results if r[1] is not None]
    cnts = [r[2] for r in results if r[2] is not None]
    rows = [r[3] for r in results if r[3] is not None]
    errors = {r[0]: r[4] for r in results if r[4]}
    X = tocf("extract")
    segments = X.segments_frame(pd.concat(segs, ignore_index=True).to_dict("records")) if segs else X.segments_frame([])
    segments = segments.sort_values(["route_uid", "seg"], kind="stable").reset_index(drop=True)
    counts = pd.concat(cnts, ignore_index=True) if cnts else pd.DataFrame(columns=SEGMENT_COUNTS_COLUMNS)
    counts = counts[SEGMENT_COUNTS_COLUMNS].sort_values(["route_uid", "seg"], kind="stable").reset_index(drop=True)
    routes_meta = pd.DataFrame(rows, columns=ROUTES_META_COLUMNS)
    return {"segments": segments, "counts": counts, "routes_meta": routes_meta, "errors": errors}


def _nice():
    import os
    try:
        os.nice(10)
    except OSError:
        pass


def summary(segments: pd.DataFrame, routes_meta: pd.DataFrame) -> dict:
    """Counts for the console: routes, segments, src_kind shares, truncated / read-error segments, rate medians."""
    n = len(segments)
    kinds = segments["src_kind"].value_counts().to_dict() if n else {}
    out = {"routes": int(len(routes_meta)), "segments": int(n),
           "src_kind": {k: {"n": int(v), "share": round(v / n, 4)} for k, v in kinds.items()},
           "segments_truncated": int(routes_meta["n_truncated"].sum()) if len(routes_meta) else 0,
           "segments_read_error": int(routes_meta["n_read_error"].sum()) if len(routes_meta) else 0,
           "routes_with_truncated": int((routes_meta["n_truncated"] > 0).sum()) if len(routes_meta) else 0}
    for col in SEG_RATE_TOPICS:
        v = pd.to_numeric(segments[col], errors="coerce") if n else pd.Series(dtype=float)
        out[f"{col}_median"] = float(np.nanmedian(v)) if v.notna().any() else None
    return out
