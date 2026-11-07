#!/usr/bin/env python3
"""S3a (1): fingerprint -> upstream opendbc platform mapping and the platform_specs.csv skeleton.

Reads an opendbc checkout (config.OPENDBC_ROOT, else the local fallback), the distinct `fingerprint` spellings of
TAXONOMY_DIR/csv/routes_taxonomy.csv (that column plus dur_min / model / source / car_dir only; never dongle or rel_path),
the distinct car-model folder spellings (`car_dir`) of its directory / fw-match rows (the labels stage falls back to the
folder name for these mock / unreadable fingerprints, so the spelling needs a row like a fingerprint) and the car_fp
values of the private route probe. Writes vtt/specs_data/fingerprint_to_platform.csv (sorted; hand-resolved manual rows are
kept) and merges a skeleton into vtt/specs_data/platform_specs.csv (filled cells are never overwritten). Prints counts only.
Exit 2 when a taxonomy fingerprint or folder spelling ends up without a row or the opendbc checkout is missing.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from vtt import config as C
from vtt import labels as L
from vtt import specs as S
from vtt.paths import P
from vtt.private import mask


FINGERPRINT_SOURCES = ("fingerprint", "alias", "prefix")
FOLDER_SOURCES = ("directory", "fw-match")        # taxonomy rows whose labels came from the folder name / firmware match
FOLDER_SKIP = ("UNKNOWN", "MOCK")                 # folder spellings that carry no car information
TAX_COLS = ("fingerprint", "model", "dur_min", "source", "car_dir")


def taxonomy_csv_default() -> Path:
    return C.TAXONOMY_DIR / "csv" / "routes_taxonomy.csv"


def load_taxonomy_fingerprints(path: Path) -> pd.DataFrame:
    """fingerprint / model / dur_min / source / car_dir columns of routes_taxonomy.csv (nothing else is read)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=list(TAX_COLS))
    cols = pd.read_csv(path, nrows=0).columns
    use = [c for c in TAX_COLS if c in cols]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    for c in TAX_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def folder_spellings(tax: pd.DataFrame) -> list[str]:
    """Distinct car_dir spellings of the directory / fw-match taxonomy rows, Unknown and MOCK excluded (sorted)."""
    if not len(tax):
        return []
    t = tax[tax["source"].isin(FOLDER_SOURCES)]
    return sorted({str(x).strip() for x in t["car_dir"].dropna()
                   if str(x).strip() and S.normalize(x) not in FOLDER_SKIP})


def probe_fingerprints(data_root: Path) -> list[str]:
    """car_fp values of private/route_probe.parquet (and the seed copy) plus routes_resolved.parquet when present."""
    out: set[str] = set()
    for p in (L.rebase(P.ROUTE_PROBE, data_root), L.rebase(P.SEEDS, data_root) / "route_probe.parquet",
              L.rebase(P.ROUTES_RESOLVED, data_root)):
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["car_fp"])
        except Exception:        # noqa: BLE001  column missing or unreadable
            continue
        out |= {str(x).strip() for x in df["car_fp"].dropna() if str(x).strip()}
    return sorted(out)


def platform_models(mapping: pd.DataFrame, tax: pd.DataFrame, tables) -> dict[str, tuple[str, str, str, str]]:
    """platform -> (brand, model, body, size_code): majority model of the fingerprint-labelled taxonomy routes whose
    fingerprint maps to the platform, else the resolved model of its fingerprint spellings; empty when unknown."""
    fp2plat = dict(zip(mapping["fingerprint"], mapping["platform"]))
    votes: dict[str, Counter] = defaultdict(Counter)
    if len(tax):
        t = tax.dropna(subset=["fingerprint", "model"])
        t = t[t["source"].isin(FINGERPRINT_SOURCES)]        # folder-derived models (mock fingerprints) do not vote
        for fp, mdl in zip(t["fingerprint"], t["model"]):
            plat = fp2plat.get(str(fp).strip())
            if plat and isinstance(mdl, str) and mdl and mdl != "Unknown":
                votes[plat][mdl] += 1
    out: dict[str, tuple[str, str, str, str]] = {}
    for plat, fps in mapping.groupby("platform")["fingerprint"]:
        if votes.get(plat):
            mdl = votes[plat].most_common(1)[0][0]
        else:
            c = Counter()
            for fp in fps:
                rf = L.resolve_full(fp, tables)
                if rf["model"] != "Unknown":
                    c[rf["model"]] += 1
            mdl = c.most_common(1)[0][0] if c else "Unknown"
        if mdl == "Unknown":                      # leave the cells empty so a later run or the user can fill them
            out[plat] = ("", "", "", "")
            continue
        brand, body, size = L.model_attrs(mdl, tables)
        out[plat] = (brand, mdl, body, size)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--opendbc-root", type=Path, default=None, help="opendbc checkout (default: config rule)")
    ap.add_argument("--taxonomy-csv", type=Path, default=taxonomy_csv_default())
    ap.add_argument("--taxonomy-dir", type=Path, default=C.TAXONOMY_DIR, help="folder holding taxonomy_tables.py")
    ap.add_argument("--specs-dir", type=Path, default=S.SPECS_DIR, help="where the two CSVs live (default vtt/specs_data)")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    a = ap.parse_args(argv)

    root = Path(a.opendbc_root) if a.opendbc_root else S.opendbc_root()
    if not (root / "opendbc" / "car").is_dir():
        print(f"opendbc checkout not found under {mask(str(root))}; clone it to {mask(str(C.OPENDBC_ROOT))}")
        return 2
    platforms = S.parse_platforms(root)
    migration = S.parse_migration(root)
    print(f"opendbc: {len(platforms)} platforms ({int(platforms.wheelbase_mm.notna().sum())} with CarSpecs), "
          f"{len(migration)} migration names, root={'upstream' if root == C.OPENDBC_ROOT else 'local fallback'}")

    tax = load_taxonomy_fingerprints(a.taxonomy_csv)
    tax_fps = sorted({str(x).strip() for x in tax["fingerprint"].dropna() if str(x).strip()})
    dir_fps = folder_spellings(tax)
    probe_fps = probe_fingerprints(a.data_root)
    fp_csv = Path(a.specs_dir) / S.FP2PLATFORM_CSV.name
    specs_csv = Path(a.specs_dir) / S.PLATFORM_SPECS_CSV.name
    existing = S.read_fp2platform(fp_csv)
    all_fps = sorted(set(tax_fps) | set(dir_fps) | set(probe_fps) | set(existing["fingerprint"]))
    mapping = S.map_fingerprints(all_fps, platforms, migration, existing)
    n_dir_new = len(set(dir_fps) - set(tax_fps) - set(probe_fps) - set(existing["fingerprint"]))
    print(f"fingerprints: taxonomy {len(tax_fps)}, probe {len(probe_fps)}, folder spellings {len(dir_fps)} ({n_dir_new} not seen "
          f"as a fingerprint), existing rows {len(existing)}, union {len(mapping)}")
    counts = mapping["method"].value_counts().to_dict()
    print("method counts: " + ", ".join(f"{m}={counts.get(m, 0)}" for m in S.METHODS))
    hours = tax.dropna(subset=["fingerprint"]).groupby("fingerprint")["dur_min"].sum() / 60.0 if len(tax) else pd.Series(dtype=float)
    if len(hours):
        mh = mapping.set_index("fingerprint")["method"]
        by_m = hours.groupby(hours.index.map(mh)).sum()
        tot = float(hours.sum()) or 1.0
        print("hours by method: " + ", ".join(f"{m}={by_m.get(m, 0.0):.0f} h ({100 * by_m.get(m, 0.0) / tot:.1f} %)" for m in S.METHODS))
    unresolved = mapping[(mapping["method"] == "manual") & mapping["note"].str.startswith(S.UNRESOLVED)]
    print(f"unresolved spellings: {len(unresolved)} (edit platform/note in {fp_csv.name} to resolve by hand)")

    tables = L.taxonomy_tables(a.taxonomy_dir)
    pm = platform_models(mapping, tax, tables)
    skeleton = S.skeleton_specs(mapping, platforms, pm)
    merged = S.merge_specs(S.read_platform_specs_raw(specs_csv) if specs_csv.exists() else None, skeleton)
    n_wb = int((merged["wheelbase_mm"].astype(str).str.strip() != "").sum())
    n_tr = int((merged["track_front_mm"].astype(str).str.strip() != "").sum())
    print(f"platform_specs: {len(merged)} platforms, {n_wb} with wheelbase, {n_tr} with track, "
          f"confidence {merged['spec_confidence'].value_counts().to_dict()}")

    missing = sorted((set(tax_fps) | set(dir_fps)) - set(mapping["fingerprint"]))
    if missing:
        print(f"ACCEPTANCE FAILED: {len(missing)} taxonomy fingerprints / folder spellings without a mapping row")
        return 2
    if a.dry_run:
        print("dry-run: nothing written")
        return 0
    S.write_csv_sorted(mapping, fp_csv)
    S.write_csv_sorted(merged, specs_csv)
    print(f"wrote {fp_csv.name} ({len(mapping)} rows) and {specs_csv.name} ({len(merged)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
