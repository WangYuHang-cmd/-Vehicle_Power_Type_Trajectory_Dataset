#!/usr/bin/env python3
"""S3a (1b): fill the empty cells of vtt/specs_data/platform_specs.csv from the vehicle-spec research tables.

Sources, in order of preference for every platform row (filled cells are never overwritten):
  1. research rows of the fingerprint spellings that map to the platform (vtt/specs_data/fingerprint_to_platform.csv,
     spellings normalised like the research keys: upper case, fork suffixes stripped);
  2. research rows of the same taxonomy car model (user ruling of 5 Nov 2026: specs are taken per car model, generations
     within a model only refine the value when the spelling is known);
  3. when the tracks are still missing but the overall width is known: mean track = TRACK_WIDTH_RATIO x width
     (ratio measured on the rows that have both: median 0.86, p10-p90 0.846-0.868), flagged in track_source.
Values are medians over the candidate rows after preferring the better confidence (A < B < C).
Inputs: vtt/specs_data/specs_research_merged.csv (passes 1-3 merged) and optional specs_fill_pass4.csv (Wikipedia pass).
Prints counts only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtt import specs as S          # noqa: E402

TRACK_WIDTH_RATIO = 0.86
NUM = ["wheelbase_mm", "track_front_mm", "track_rear_mm", "curb_mass_kg", "curb_mass_min_kg", "curb_mass_max_kg", "width_mm"]
SUF = re.compile(r"(_MMR(_\d{4})?|_PEDAL|_ASCM|_NON_SCC|_NO_ACC|_NONSCC|_MOCK)+$")


def norm_key(fp) -> str:
    s = str(fp).strip().upper().replace(" ", "_")
    s = re.sub(r"[^A-Z0-9_]", "_", s)
    return SUF.sub("", s)


def load_research(d: Path) -> pd.DataFrame:
    r = pd.read_csv(d / "specs_research_merged.csv")
    r["variant"] = r.get("variant", pd.Series([""] * len(r))).fillna("")
    r["conf"] = r.get("confidence", "C").fillna("C").astype(str).str[:1]
    p4 = d / "specs_fill_pass4.csv"
    if p4.exists():
        f = pd.read_csv(p4)
        f = f.rename(columns={"spellings_str": "keys_str"})
        rows = []
        for rec in f.to_dict("records"):
            keys = str(rec.get("keys_str") or "").split()
            for k in keys or [""]:
                rows.append({"platform_key": k, "model": rec.get("model"), "brand": rec.get("brand"), "variant": rec.get("variant") or "",
                             "body": rec.get("body"), "wheelbase_mm": rec.get("wheelbase_mm"), "track_front_mm": rec.get("track_front_mm"),
                             "track_rear_mm": rec.get("track_rear_mm"), "curb_mass_kg": rec.get("curb_mass_kg"),
                             "curb_mass_min_kg": rec.get("curb_mass_min_kg"), "curb_mass_max_kg": rec.get("curb_mass_max_kg"),
                             "width_mm": rec.get("width_mm"), "conf": str(rec.get("confidence") or "C")[:1],
                             "source_urls": rec.get("source_urls") or "", "fill_source": "pass4"})
        r = pd.concat([r, pd.DataFrame(rows)], ignore_index=True)
    for c in NUM:
        r[c] = pd.to_numeric(r.get(c), errors="coerce")
    r["platform_key"] = r["platform_key"].astype(str).map(norm_key)
    r["source_urls"] = r.get("source_urls", "").fillna("").astype(str)
    return r


def pick(rows: pd.DataFrame, col: str):
    """Median of `col` over the best-confidence rows that have it; (value, confidence, sources) or (nan, None, '')."""
    have = rows[rows[col].notna()]
    if have.empty:
        return np.nan, None, ""
    best = have.conf.min()
    sel = have[have.conf == best]
    urls = " ".join(sorted({u for s in sel.source_urls for u in str(s).split() if u}))[:600]
    return float(sel[col].median()), best, urls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs-dir", type=Path, default=S.SPECS_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.specs_dir)
    specs = S.read_platform_specs_raw(d / "platform_specs.csv")
    fp2 = pd.read_csv(d / "fingerprint_to_platform.csv", dtype=str).fillna("")
    research = load_research(d)
    keys_of: dict[str, set[str]] = {}
    for fp, plat in zip(fp2.fingerprint, fp2.platform):
        if plat:
            keys_of.setdefault(plat, set()).add(norm_key(fp))
            keys_of[plat].add(norm_key(plat))
    by_key = {k: g for k, g in research.groupby("platform_key")}
    by_model = {m: g for m, g in research.groupby("model")}
    n_fill = {c: 0 for c in ("wheelbase_mm", "track_front_mm", "track_rear_mm", "curb_mass_kg", "track_est")}
    out = specs.copy()
    for i, row in out.iterrows():
        plat = str(row["platform"]); model = str(row.get("model") or "")
        cand_keys = keys_of.get(plat, set()) | {norm_key(plat)}
        plat_rows = pd.concat([by_key[k] for k in cand_keys if k in by_key], ignore_index=True) if any(k in by_key for k in cand_keys) else research.iloc[0:0]
        model_rows = by_model.get(model, research.iloc[0:0]) if model else research.iloc[0:0]
        notes, confs, urls_all = [], [], []
        for col, src_col, tag in (("wheelbase_mm", "wheelbase_source", "wb"), ("track_front_mm", "track_source", "tr"),
                                   ("track_rear_mm", "track_source", "tr"), ("curb_mass_kg", "mass_source", "mass")):
            if str(row.get(col) or "").strip():
                continue
            v, conf, urls = pick(plat_rows, col)
            level = "spelling"
            if not np.isfinite(v):
                v, conf, urls = pick(model_rows, col)
                level = "model"
            if np.isfinite(v):
                out.at[i, col] = f"{v:.0f}"
                if not str(row.get(src_col) or "").strip() or out.at[i, src_col] in ("", "research:model", "research:spelling"):
                    out.at[i, src_col] = f"research:{level}"
                confs.append(conf or "C"); urls_all.append(urls); n_fill[col] += 1
        # tracks still missing -> estimate from overall width
        if not str(out.at[i, "track_front_mm"]).strip() or not str(out.at[i, "track_rear_mm"]).strip():
            w, conf, urls = pick(plat_rows, "width_mm")
            if not np.isfinite(w):
                w, conf, urls = pick(model_rows, "width_mm")
            if np.isfinite(w):
                est = f"{TRACK_WIDTH_RATIO * w:.0f}"
                if not str(out.at[i, "track_front_mm"]).strip():
                    out.at[i, "track_front_mm"] = est
                if not str(out.at[i, "track_rear_mm"]).strip():
                    out.at[i, "track_rear_mm"] = est
                out.at[i, "track_source"] = f"est_{TRACK_WIDTH_RATIO}xwidth"
                notes.append(f"tracks estimated as {TRACK_WIDTH_RATIO} x overall width {w:.0f} mm"); confs.append("C"); urls_all.append(urls)
                n_fill["track_est"] += 1
        if confs:
            best = min(confs)
            cur = str(out.at[i, "spec_confidence"] or "").strip()
            out.at[i, "spec_confidence"] = best if (not cur or cur > best) else cur
            urls = " ".join(sorted({u for s in urls_all for u in str(s).split() if u}))[:800]
            if urls and not str(out.at[i, "source_urls"] or "").strip():
                out.at[i, "source_urls"] = urls
            if notes:
                prev = str(out.at[i, "note"] or "").strip()
                out.at[i, "note"] = (prev + "; " if prev else "") + "; ".join(notes)
    wb = (out.wheelbase_mm.astype(str).str.strip() != "").sum(); tr = ((out.track_front_mm.astype(str).str.strip() != "") & (out.track_rear_mm.astype(str).str.strip() != "")).sum()
    ms = (out.curb_mass_kg.astype(str).str.strip() != "").sum()
    print(f"platform_specs: {len(out)} rows; filled now: {n_fill}; coverage after fill: wheelbase {wb}, both tracks {tr}, mass {ms}; "
          f"confidence {out.spec_confidence.replace('', 'none').value_counts().to_dict()}")
    if not a.dry_run:
        tmp = d / ".platform_specs.csv.tmp"
        out.to_csv(tmp, index=False)
        tmp.replace(d / "platform_specs.csv")
        print("wrote platform_specs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
