"""S2d: acceptance of the extraction per software era.

What is judged (plan S2 acceptance): among FULL qlog segments (span >= FULL_SEG_SPAN_S; the trailing partial minute of a route
is reported separately, never judged) the share with >= MIN_CARSTATE_ROWS carState rows and >= MIN_RADAR_ROWS radarState
rows must reach ERA_GOOD_MIN in every era that has full segments (a radar count below MIN_RADAR_ROWS is only reported: several
device/software generations log radarState at 2-3 Hz; fewer than MIN_RADAR_PRESENT_ROWS rows counts as radar-absent and is judged); routes expected (chunk lists, else uid_map) but absent from
work/signals -- failed or never extracted -- must stay below FAIL_MAX_FRAC; every route_meta must carry
config.EXTRACTOR_VERSION. The per-segment unknown-union histogram per era makes a silently dropped topic of a new writer
visible. Inputs are the three S2c tables (vtt.segments). Units: spans s, shares 0..1.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt.private import mask

MIN_CARSTATE_ROWS = 500          # rows per full qlog segment (10 Hz x 60 s = 600 nominal)
MIN_RADAR_ROWS = 150             # informational: rows per full qlog segment (4 Hz x 60 s = 240 nominal; some devices log 2-3 Hz)
MIN_RADAR_PRESENT_ROWS = 60      # hard: a full segment with fewer radarState rows (< 1 Hz) counts as radar-absent
FULL_SEG_SPAN_S = 55.0           # a segment shorter than this is the route's partial last minute (or a cut file): not judged
ERA_GOOD_MIN = 0.98              # per judged era; the catch-all era "other" (unparsed fork version strings) is reported, not judged
ERA_UNJUDGED = ("other",)
FAIL_MAX_FRAC = 0.005
UNKNOWN_BINS = ((0, 0, "unk_0"), (1, 10, "unk_1_10"), (11, 100, "unk_11_100"), (101, 300, "unk_101_300"), (301, math.inf, "unk_301p"))
ERA_COLUMNS = ["software_era", "n_routes", "n_segments", "n_qlog", "n_rlog_dec", "n_none", "n_short", "n_full", "n_good", "good_frac",
               "n_low_carstate", "n_low_radar", "n_radar_absent", "carstate_rows_med", "radar_rows_med", "n_truncated", "n_read_error",
               "n_version_mismatch", "unk_median", "unk_max"] + [b[2] for b in UNKNOWN_BINS]


# ----------------------------------------------------------------------------- per-era table
def _era_row(era: str, seg: pd.DataFrame, routes: pd.DataFrame) -> dict:
    qlog = seg[seg["src_kind"] == "qlog"]
    span = pd.to_numeric(qlog["span_s"], errors="coerce")
    full = qlog[span >= FULL_SEG_SPAN_S]
    low_cs = full["n_carState"] < MIN_CARSTATE_ROWS
    low_rs = full["n_radarState"] < MIN_RADAR_ROWS                 # reported only (slow radar writers are a known property)
    absent_rs = full["n_radarState"] < MIN_RADAR_PRESENT_ROWS      # judged: radar stream effectively missing
    good = full[~low_cs & ~absent_rs]
    unk = pd.to_numeric(qlog["unknown_union_n"], errors="coerce").fillna(0).to_numpy()
    row = {"software_era": era, "n_routes": int(len(routes)), "n_segments": int(len(seg)), "n_qlog": int(len(qlog)),
           "n_rlog_dec": int((seg["src_kind"] == "rlog_dec").sum()), "n_none": int((seg["src_kind"] == "none").sum()),
           "n_short": int((span < FULL_SEG_SPAN_S).sum()), "n_full": int(len(full)), "n_good": int(len(good)),
           "good_frac": (len(good) / len(full)) if len(full) else np.nan,
           "n_low_carstate": int(low_cs.sum()), "n_low_radar": int(low_rs.sum()), "n_radar_absent": int(absent_rs.sum()),
           "carstate_rows_med": float(full["n_carState"].median()) if len(full) else np.nan,
           "radar_rows_med": float(full["n_radarState"].median()) if len(full) else np.nan,
           "n_truncated": int(routes["n_truncated"].sum()), "n_read_error": int(routes["n_read_error"].sum()),
           "n_version_mismatch": int((routes["extractor_version"].astype(str) != C.EXTRACTOR_VERSION).sum()),
           "unk_median": float(np.median(unk)) if len(unk) else np.nan, "unk_max": int(unk.max()) if len(unk) else 0}
    for lo, hi, name in UNKNOWN_BINS:
        row[name] = int(((unk >= lo) & (unk <= hi)).sum())
    return row


def eras_present(routes_meta: pd.DataFrame) -> list[str]:
    """Every software_era value of routes_meta, config.SOFTWARE_ERAS order first, then any other value sorted (a table written
    before the era normalisation may carry one; it is judged like the others, never silently skipped)."""
    present = set(routes_meta["software_era"].fillna("other").astype(str))
    return [e for e in C.SOFTWARE_ERAS if e in present] + sorted(present - set(C.SOFTWARE_ERAS))


def era_table(segments: pd.DataFrame, counts: pd.DataFrame, routes_meta: pd.DataFrame) -> pd.DataFrame:
    """One row per software_era present (eras_present) plus an 'ALL' row (ERA_COLUMNS); segments and counts are joined on
    (route_uid, seg)."""
    routes_meta = routes_meta.assign(software_era=routes_meta["software_era"].fillna("other").astype(str))
    seg = counts.merge(segments[["route_uid", "seg", "unknown_union_n"]], on=["route_uid", "seg"], how="left")
    seg = seg.merge(routes_meta[["route_uid", "software_era"]], on="route_uid", how="left")
    seg["software_era"] = seg["software_era"].fillna("other")
    rows = []
    for era in eras_present(routes_meta):
        r = routes_meta[routes_meta["software_era"] == era]
        rows.append(_era_row(era, seg[seg["software_era"] == era], r))
    rows.append(_era_row("ALL", seg, routes_meta))
    return pd.DataFrame(rows, columns=ERA_COLUMNS)


# ----------------------------------------------------------------------------- coverage and failures
def expected_uids(chunks_dir: Path, uid_map_path: Path) -> tuple[set[str], str]:
    """(route_uids the extraction should have produced, source): the union of build/chunks/*.csv when present, else uid_map."""
    chunks_dir, uid_map_path = Path(chunks_dir), Path(uid_map_path)
    files = sorted(chunks_dir.glob("routes_chunk_*.csv")) if chunks_dir.exists() else []
    if files:
        uids: set[str] = set()
        for f in files:
            uids |= set(pd.read_csv(f, dtype=str)["route_uid"].astype(str).str.strip())
        return uids, f"{len(files)} chunk files"
    if uid_map_path.exists():
        return set(pd.read_parquet(uid_map_path, columns=["route_uid"])["route_uid"].astype(str)), "uid_map"
    return set(), "none"


def failures_table(failures_csv: Path, extracted: set[str]) -> dict:
    """Counts of work/logs/s02_failures.csv: rows, distinct routes, routes still absent (unresolved) and a status histogram
    (identifier-free: status is 'error:<Type>', the message column is never returned)."""
    p = Path(failures_csv)
    if not p.exists():
        return {"n_rows": 0, "n_uids": 0, "n_unresolved": 0, "status_hist": {}}
    f = pd.read_csv(p, dtype=str)
    uids = set(f["route_uid"].astype(str)) if "route_uid" in f else set()
    hist = f["status"].astype(str).map(mask).value_counts().to_dict() if "status" in f else {}
    return {"n_rows": int(len(f)), "n_uids": len(uids), "n_unresolved": len(uids - extracted), "status_hist": {str(k): int(v) for k, v in hist.items()}}


def evaluate(era: pd.DataFrame, n_expected: int, n_missing: int) -> tuple[bool, list[str]]:
    """Acceptance verdict and its reasons (empty when accepted)."""
    reasons = []
    for _, r in era[era["software_era"] != "ALL"].iterrows():
        if str(r["software_era"]) in ERA_UNJUDGED:
            continue
        if r["n_full"] > 0 and r["good_frac"] < ERA_GOOD_MIN:
            reasons.append(f"era {r['software_era']}: good full-segment share {r['good_frac']:.4f} < {ERA_GOOD_MIN} ({int(r['n_good'])}/{int(r['n_full'])})")
    if n_expected:
        frac = n_missing / n_expected
        if frac > FAIL_MAX_FRAC:
            reasons.append(f"routes missing from work/signals (failed or not extracted): {n_missing}/{n_expected} = {frac:.4f} > {FAIL_MAX_FRAC}")
    else:
        reasons.append("no expected route list (no chunk files and no uid_map)")
    mism = int(era.loc[era["software_era"] == "ALL", "n_version_mismatch"].sum()) if len(era) else 0
    if mism:
        reasons.append(f"{mism} routes with extractor_version != {C.EXTRACTOR_VERSION}")
    return (not reasons), reasons


# ----------------------------------------------------------------------------- report
def _fmt(v) -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else (f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
    return str(v)


def render_md(era: pd.DataFrame, ok: bool, reasons: list[str], coverage: dict, fails: dict) -> str:
    lines = ["# S2 extraction acceptance", "", f"Verdict: **{'ACCEPTED' if ok else 'REJECTED'}**", ""]
    lines += [f"- {mask(r)}" for r in reasons] or ["- all checks passed"]
    lines += ["", "## Coverage", f"- expected routes: {coverage['n_expected']} (source: {coverage['source']})",
              f"- extracted routes (route_meta.json present): {coverage['n_extracted']}",
              f"- expected but missing: {coverage['n_missing']}", f"- extracted but not expected: {coverage['n_unexpected']}",
              f"- uid_map routes without route_meta: {coverage['n_uidmap_missing']}",
              f"- failures.csv rows: {fails['n_rows']} (distinct routes {fails['n_uids']}, still missing {fails['n_unresolved']})"]
    if fails["status_hist"]:
        lines += ["- failure statuses: " + ", ".join(f"{k} x{v}" for k, v in sorted(fails["status_hist"].items()))]
    lines += ["", "## Thresholds", f"- full segment: qlog source and span >= {FULL_SEG_SPAN_S} s",
              f"- good segment: >= {MIN_CARSTATE_ROWS} carState rows and >= {MIN_RADAR_PRESENT_ROWS} radarState rows (< {MIN_RADAR_ROWS} radar rows is reported as 'low radar', not judged)",
              f"- per era: good share >= {ERA_GOOD_MIN}; missing routes <= {FAIL_MAX_FRAC}; extractor_version == {C.EXTRACTOR_VERSION}",
              "", "## Per software era", ""]
    cols = list(era.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in era.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    lines += ["", "unknown-union bins count qlog segments by unknown_union_n (members the parse schema does not know)."]
    return "\n".join(lines) + "\n"
