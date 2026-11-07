#!/usr/bin/env python3
"""S3a (3): per-route labels -> build/routes_labels.parquet (public-safe) and private/labels_private.parquet.

Joins private/routes_resolved.parquet to TAXONOMY_DIR/csv/routes_taxonomy.csv on the chosen copy's rel_path
(fallback: any copy's rel_path; fallback: resolve(car_fp) + ECU rules -> label_source 'resolved_new'), takes
region / in_house / country from ROUTE_MASTER_CSV (never lat/lon), re-applies the ECU rules from
private/route_audit.parquet to platforms still MIXED, maps fingerprints to platforms (vtt/specs_data/fingerprint_to_platform.csv;
mock / unreadable fingerprints fall back to the folder spelling, then to the unique platform of the taxonomy model)
and merges vtt/specs_data/platform_specs.csv. build/label_overrides_routes.csv (route_uid, field, value, reason) is applied
last; the S7 file build/label_overrides.csv (scope, key, ...) is a different table and is never read here.
Prints the size_class x powertrain cell table and counts only. Exit 2 on failed acceptance.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt import labels as L
from vtt import specs as S
from vtt.paths import P
from vtt.private import mask, write_parquet_atomic

PUBLIC_COLUMNS = ["route_uid", "platform", "brand", "model", "body", "size_code", "powertrain_raw", "powertrain",
                  "label_source", "label_grade", "label_note", "region", "software_era", "wheelbase_mm", "track_front_mm",
                  "track_rear_mm", "footprint_m2", "curb_mass_kg", "spec_confidence", "size_class"]
PRIVATE_EXTRA = ["rel_path", "dongle_log", "fingerprint_raw", "in_house", "country"]
OVERRIDE_COLUMNS = ("route_uid", "field", "value", "reason")           # reason may be absent
OVERRIDE_FIELDS = ("platform", "brand", "model", "body", "size_code", "powertrain_raw", "powertrain", "label_note", "region",
                   "wheelbase_mm", "track_front_mm", "track_rear_mm", "curb_mass_kg", "spec_confidence", "label_source")
NUMERIC_FIELDS = ("wheelbase_mm", "track_front_mm", "track_rear_mm", "curb_mass_kg")
SPEC_FIELDS = ["wheelbase_mm", "track_front_mm", "track_rear_mm", "footprint_m2", "curb_mass_kg", "spec_confidence"]
TAX_COLS = ["rel_path", "car_dir", "fingerprint", "model", "brand", "body", "size", "pt_platform", "powertrain", "source", "dur_min"]
MASTER_COLS = ["rel_path", "region", "country", "in_house"]
MOCK_FP = "MOCK"                       # openpilot's placeholder fingerprint: no car information
FOLDER_SOURCES = ("directory", "fw-match")      # taxonomy rows labelled without a usable fingerprint
FOLDER_SKIP = (MOCK_FP, "UNKNOWN")


# ------------------------------------------------------------------------------------------------------- loaders
def load_taxonomy(path: Path) -> pd.DataFrame:
    """routes_taxonomy.csv restricted to the label columns (dongle, miles, group, lat/lon never read)."""
    cols = pd.read_csv(path, nrows=0).columns
    use = [c for c in TAX_COLS if c in cols]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    for c in TAX_COLS:
        if c not in df.columns:
            df[c] = None
    return df.drop_duplicates("rel_path")


def load_master(path: Path | None) -> pd.DataFrame:
    """route_master.csv: rel_path, region, country, in_house only."""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=MASTER_COLS)
    cols = pd.read_csv(path, nrows=0).columns
    use = [c for c in MASTER_COLS if c in cols]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    for c in MASTER_COLS:
        if c not in df.columns:
            df[c] = None
    return df.drop_duplicates("rel_path")


def load_audit(path: Path) -> pd.DataFrame:
    """private/route_audit.parquet: route_uid, ecus, transmission (empty frame when absent)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["route_uid", "ecus", "transmission"])
    df = pd.read_parquet(path)
    for c in ("ecus", "transmission"):
        if c not in df.columns:
            df[c] = None
    return df[["route_uid", "ecus", "transmission"]].drop_duplicates("route_uid")


def read_overrides(path: Path) -> pd.DataFrame | None:
    """build/label_overrides_routes.csv as strings (None when absent). ValueError with a counts-only message when the
    file does not carry the per-route schema (e.g. the S7 drive / vehicle / platform override table)."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in OVERRIDE_COLUMNS[:3] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: {len(missing)} required columns missing ({', '.join(missing)}); expected "
                         f"{', '.join(OVERRIDE_COLUMNS)} - the S7 table build/label_overrides.csv has another schema")
    if "reason" not in df.columns:
        df["reason"] = ""
    return df[list(OVERRIDE_COLUMNS)]


def lookup_by_copy(rel_path, copies_rel, index: dict[str, int]) -> tuple[int | None, bool]:
    """Row index for the chosen copy's rel_path, else for any other copy (flagged True)."""
    if isinstance(rel_path, str) and rel_path in index:
        return index[rel_path], False
    for c in L.parse_copies(copies_rel):
        if c in index:
            return index[c], True
    return None, False


# ------------------------------------------------------------------------------------------------------- builder
def _folder_platform(car_dir, fp2plat: dict[str, str]) -> tuple[str | None, bool]:
    """(platform, usable) for a car-model folder spelling; usable is False for Unknown / MOCK / empty folders."""
    if not isinstance(car_dir, str) or not car_dir.strip() or S.normalize(car_dir) in FOLDER_SKIP:
        return None, False
    return fp2plat.get(car_dir.strip()) or fp2plat.get(S.normalize(car_dir)), True


def label_route(r, tax: pd.DataFrame, tax_index: dict[str, int], audit: dict[str, tuple], fp2plat: dict[str, str],
                tables, missing_fp: set[str], model_index: dict[str, list[str]] | None = None,
                missing_dir: set[str] | None = None) -> dict:
    """Label dict of one routes_resolved row (see module doc for the fallback order)."""
    notes: list[str] = []
    folder_missing = False                 # folder spelling without a mapping row (reported by build_labels)
    model_index = model_index if model_index is not None else {}
    ti, other_copy = lookup_by_copy(r.rel_path, getattr(r, "copies_rel", None), tax_index)
    car_fp = r.car_fp if isinstance(r.car_fp, str) and r.car_fp.strip() else None
    if ti is not None:
        t = tax.iloc[ti]
        fp_raw = t.fingerprint if isinstance(t.fingerprint, str) and t.fingerprint.strip() else car_fp
        model, brand, body, size_code = t.model, t.brand, t.body, t["size"]
        pt_platform, pt_raw, source = t.pt_platform, t.powertrain, t.source
        if other_copy:
            notes.append("taxonomy row taken from another copy of the route")
        if isinstance(fp_raw, str):
            dn = L.dir_note(fp_raw, tables)
            if dn:
                notes.append(dn)
        if pt_platform == "MIXED" and pt_raw not in (None, "MIXED") and isinstance(pt_raw, str):
            notes.append(f"MIXED platform resolved to {pt_raw} per route (ECU list)")
    else:
        rf = L.resolve_full(car_fp, tables)
        fp_raw = car_fp
        model, brand, body, size_code = rf["model"], rf["brand"], rf["body"], rf["size_code"]
        pt_platform = pt_raw = rf["powertrain_raw"]
        if rf["how"] in ("fingerprint", "alias", "prefix"):
            source = "resolved_new"
            notes.append(f"resolved from the probe fingerprint ({rf['how']})")
        elif rf["how"] == "unmapped":
            source = "unmapped"
        else:
            source = "none"
        if rf["note"]:
            notes.append(rf["note"])
    if pt_raw == "MIXED" and r.route_uid in audit:
        ecus, trans = audit[r.route_uid]
        new_pt, rule = L.ecu_resolve(model, fp_raw, "MIXED", ecus, trans)
        if rule:
            pt_raw = new_pt
            notes.append(f"MIXED platform resolved to {new_pt} by {rule}")
    platform = None
    is_mock = isinstance(fp_raw, str) and S.normalize(fp_raw) == MOCK_FP
    if isinstance(fp_raw, str) and not is_mock:
        platform = fp2plat.get(fp_raw.strip()) or fp2plat.get(S.normalize(fp_raw))
        if platform is None:
            missing_fp.add(fp_raw.strip())
    elif ti is not None and source in FOLDER_SOURCES:
        # no usable fingerprint: the folder spelling (directory rows) or the unique platform of the model (fw-match rows)
        why = "mock fingerprint" if is_mock else "fingerprint unreadable"
        car_dir = tax.iloc[ti].car_dir
        folder_plat, folder_ok = _folder_platform(car_dir, fp2plat)
        if folder_ok and folder_plat is None:
            folder_missing = True
            if missing_dir is not None:
                missing_dir.add(str(car_dir).strip())
        model_plat, n_cand = L.platform_from_model(model, pt_raw, model_index, tables)
        order = ((model_plat, "model"), (folder_plat, "folder")) if source == "fw-match" else \
                ((folder_plat, "folder"), (model_plat, "model"))
        for plat, how in order:
            if plat:
                platform = plat
                notes.append(f"platform taken from the folder name ({why})" if how == "folder"
                             else f"platform taken from the taxonomy model, its only platform ({why})")
                break
        if platform is None and n_cand > 1:
            notes.append(f"platform ambiguous: {n_cand} platforms share the model ({why})")
    return dict(route_uid=r.route_uid, platform=platform, brand=brand, model=model, body=body, size_code=size_code,
                powertrain_raw=pt_raw, powertrain=L.powertrain_map(pt_raw), label_source=source,
                label_grade=L.label_grade(source), label_note="; ".join(n for n in notes if n) or None,
                fingerprint_raw=fp_raw, rel_path=r.rel_path, dongle_log=getattr(r, "dongle_log", None),
                software_era=L.software_era(getattr(r, "version", None)), _folder_missing=folder_missing)


def attach_master(df: pd.DataFrame, routes: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """region (bucket), in_house (bool) and country from route_master by chosen rel_path, else any copy."""
    idx = {rp: i for i, rp in enumerate(master["rel_path"]) if isinstance(rp, str)}
    region, in_house, country = [], [], []
    for r in routes.itertuples(index=False):
        mi, _ = lookup_by_copy(r.rel_path, getattr(r, "copies_rel", None), idx)
        if mi is None:
            region.append(L.region_bucket(None))
            in_house.append(False)
            country.append(None)
            continue
        m = master.iloc[mi]
        region.append(L.region_bucket(m.region))
        ih = m.in_house
        in_house.append(bool(ih) if isinstance(ih, (bool, np.bool_)) else str(ih).strip().lower() in ("true", "1", "yes"))
        country.append(m.country if isinstance(m.country, str) else None)
    df["region"], df["in_house"], df["country"] = region, in_house, country
    return df


def spec_rows(specs: pd.DataFrame | None) -> dict[str, dict]:
    """platform -> {spec field: value} from platform_specs (NaN / None for empty cells)."""
    out: dict[str, dict] = {}
    if specs is None or not len(specs):
        return out
    for rec in specs.drop_duplicates("platform").to_dict("records"):
        row = {}
        for c in SPEC_FIELDS:
            v = rec.get(c)
            if c == "spec_confidence":
                row[c] = v.strip() if isinstance(v, str) and v.strip() else None
            else:
                row[c] = pd.to_numeric(v, errors="coerce") if v is not None else np.nan
        out[str(rec["platform"])] = row
    return out


def attach_specs(df: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Merge platform_specs by platform; footprint from the table or from wheelbase x mean track; size_class cut."""
    cols = ["platform"] + SPEC_FIELDS
    for c in SPEC_FIELDS:
        if c in df.columns:
            df = df.drop(columns=c)
    if len(specs):
        sp = specs[[c for c in cols if c in specs.columns]].drop_duplicates("platform")
        for c in SPEC_FIELDS:
            if c not in sp.columns:
                sp[c] = np.nan
        df = df.merge(sp, on="platform", how="left")
    else:
        for c in SPEC_FIELDS:
            df[c] = np.nan
    df["spec_confidence"] = df["spec_confidence"].where(df["spec_confidence"].astype(str).str.strip().ne("") & df["spec_confidence"].notna(), None)
    df["footprint_m2"] = L.footprint_series(df).to_numpy()
    df["size_class"] = L.cut_size_classes(df["footprint_m2"]).to_numpy()
    return df


def _clean(value):
    """Override cell -> None for empty / null spellings, else the stripped string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    return None if s.lower() in ("", "null", "none", "nan") else s


def apply_overrides(df: pd.DataFrame, overrides: pd.DataFrame | None, specs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """build/label_overrides_routes.csv rows (route_uid, field, value, reason), applied per route:
    a `platform` override first re-merges that platform's platform_specs row (wheelbase / tracks / mass / confidence),
    then the other fields follow in file order (explicit spec cells win over the re-merged ones), an explicit
    `powertrain` wins whatever the row order, otherwise `powertrain` is recomputed when `powertrain_raw` changed;
    footprint / size_class / label_grade are recomputed afterwards.
    Returns (frame, counts) with counts = applied / unknown_uid / unknown_field."""
    counts = {"applied": 0, "unknown_uid": 0, "unknown_field": 0}
    if overrides is None or not len(overrides) or df["route_uid"].duplicated().any():
        return df, counts
    df = df.set_index("route_uid", drop=False)
    by_uid: dict[str, list[tuple[str, object, str]]] = {}
    for o in overrides.itertuples(index=False):
        uid, field = str(o.route_uid).strip(), str(o.field).strip()
        if field not in OVERRIDE_FIELDS:
            counts["unknown_field"] += 1
            continue
        if uid not in df.index:
            counts["unknown_uid"] += 1
            continue
        by_uid.setdefault(uid, []).append((field, o.value, getattr(o, "reason", "")))
    table = spec_rows(specs)
    touched_fp: set[str] = set()
    explicit_dims: set[str] = set()
    for uid, rows in by_uid.items():
        fields = {f for f, _, _ in rows}
        if "platform" in fields:
            plat = _clean([v for f, v, _ in rows if f == "platform"][-1])
            df.at[uid, "platform"] = plat
            spec = table.get(plat, {})
            for c in SPEC_FIELDS:
                df.at[uid, c] = spec.get(c, None if c == "spec_confidence" else np.nan)
            touched_fp.add(uid)
        for field, value, reason in rows:
            if field != "platform":
                if field in NUMERIC_FIELDS:
                    value = pd.to_numeric(_clean(value), errors="coerce")
                    touched_fp.add(uid)
                    explicit_dims.add(uid)
                else:
                    value = _clean(value)
                df.at[uid, field] = value
            note = f"override: {reason}" if isinstance(reason, str) and reason.strip() else "override"
            prev = df.at[uid, "label_note"]
            df.at[uid, "label_note"] = f"{prev}; {note}" if isinstance(prev, str) and prev else note
            counts["applied"] += 1
        if "powertrain" not in fields and "powertrain_raw" in fields:
            df.at[uid, "powertrain"] = L.powertrain_map(df.at[uid, "powertrain_raw"])
    for uid in sorted(touched_fp):
        fp = L.footprint_m2(df.at[uid, "wheelbase_mm"], df.at[uid, "track_front_mm"], df.at[uid, "track_rear_mm"])
        given = pd.to_numeric(df.at[uid, "footprint_m2"], errors="coerce")
        if math.isnan(fp) and uid not in explicit_dims and not pd.isna(given):
            fp = float(given)                        # re-merged platform with a table footprint but no dimensions
        df.at[uid, "footprint_m2"] = fp
        df.at[uid, "size_class"] = L.size_class(fp)
    for uid in df.index:
        df.at[uid, "label_grade"] = L.label_grade(df.at[uid, "label_source"])
    return df.reset_index(drop=True), counts


def build_labels(routes: pd.DataFrame, tax: pd.DataFrame, master: pd.DataFrame, audit: pd.DataFrame, fp2plat: dict[str, str],
                 specs: pd.DataFrame, tables, overrides: pd.DataFrame | None = None,
                 merges: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Full label frame (public + private columns) and a stats dict (missing fingerprints / folder spellings, override counts)."""
    tax_index = {rp: i for i, rp in enumerate(tax["rel_path"]) if isinstance(rp, str)}
    audit_d = {str(u): (e, t) for u, e, t in zip(audit["route_uid"], audit["ecus"], audit["transmission"])}
    model_index = L.model_platform_index(specs)
    missing_fp: set[str] = set()
    missing_dir: set[str] = set()
    rows = [label_route(r, tax, tax_index, audit_d, fp2plat, tables, missing_fp, model_index, missing_dir)
            for r in routes.itertuples(index=False)]
    n_dir_routes = sum(1 for row in rows if row.pop("_folder_missing", False))
    df = pd.DataFrame(rows)
    df = attach_master(df, routes, master)
    df = attach_specs(df, specs)
    df["label_note"] = [L.label_note(p, f, n) for p, f, n in zip(df["platform"], df["fingerprint_raw"], df["label_note"])]
    df, mc = L.apply_model_merges(df, merges)
    df, oc = apply_overrides(df, overrides, specs)
    stats = {"missing_fingerprints": sorted(missing_fp), "missing_folder_spellings": sorted(missing_dir),
             "routes_missing_folder_spelling": n_dir_routes, "overrides": oc, "model_merges": mc}
    return df, stats


def cell_table_text(df: pd.DataFrame, hours: pd.Series) -> str:
    """size_class x powertrain: routes (hours)."""
    d = pd.DataFrame({"size": df["size_class"].where(df["size_class"].notna(), "null"),
                      "pt": df["powertrain"].where(df["powertrain"].notna(), "null"), "h": hours.to_numpy()})
    sizes = list(C.SIZE_CLASSES) + ["null"]
    pts = list(C.POWERTRAIN4) + ["null"]
    g = d.groupby(["size", "pt"]).agg(n=("h", "size"), h=("h", "sum"))
    lines = ["size_class x powertrain  (routes / hours)", "%-8s" % "" + "".join("%18s" % p for p in pts) + "%18s" % "total"]
    for s in sizes:
        cells = []
        for p in pts:
            n, h = (int(g.loc[(s, p), "n"]), float(g.loc[(s, p), "h"])) if (s, p) in g.index else (0, 0.0)
            cells.append("%10d / %5.0f" % (n, h))
        tot = d[d["size"] == s]
        lines.append("%-8s" % s + "".join(cells) + "%10d / %5.0f" % (len(tot), tot.h.sum()))
    return "\n".join(lines)


def acceptance(df: pd.DataFrame, routes: pd.DataFrame, stats: dict) -> list[str]:
    problems = []
    if len(df) != len(routes) or df["route_uid"].duplicated().any():
        problems.append("row count differs from routes_resolved or duplicated route_uid")
    if stats["missing_fingerprints"]:
        problems.append(f"{len(stats['missing_fingerprints'])} fingerprints without a mapping row (run scripts/s03_platforms.py)")
    if stats.get("missing_folder_spellings"):
        problems.append(f"{len(stats['missing_folder_spellings'])} folder spellings ({stats.get('routes_missing_folder_spelling', 0)} "
                        f"routes) without a mapping row (run scripts/s03_platforms.py)")
    if stats["overrides"]["unknown_field"]:
        problems.append(f"{stats['overrides']['unknown_field']} override rows name an unknown field")
    bad_pt = set(df["powertrain"].dropna()) - set(C.POWERTRAIN4)
    bad_sz = set(df["size_class"].dropna()) - set(C.SIZE_CLASSES)
    bad_rg = set(df["region"].dropna()) - (set(C.REGION_BUCKETS.values()) | {C.REGION_OTHER})
    bad_era = set(df["software_era"].dropna()) - set(C.SOFTWARE_ERAS)
    for name, bad in (("powertrain", bad_pt), ("size_class", bad_sz), ("region", bad_rg), ("software_era", bad_era)):
        if bad:
            problems.append(f"{name} outside its domain: {sorted(bad)}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--taxonomy-csv", type=Path, default=C.TAXONOMY_DIR / "csv" / "routes_taxonomy.csv")
    ap.add_argument("--taxonomy-dir", type=Path, default=C.TAXONOMY_DIR, help="folder holding taxonomy_tables.py")
    ap.add_argument("--route-master", type=Path, default=C.ROUTE_MASTER_CSV)
    ap.add_argument("--specs-dir", type=Path, default=S.SPECS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    a = ap.parse_args(argv)

    rr = L.rebase(P.ROUTES_RESOLVED, a.data_root)
    if not rr.exists():
        print(f"missing {mask(str(rr))}: run scripts/s01_probe_uidmap.py first")
        return 2
    fp_csv, specs_csv = Path(a.specs_dir) / S.FP2PLATFORM_CSV.name, Path(a.specs_dir) / S.PLATFORM_SPECS_CSV.name
    if not fp_csv.exists():
        print(f"missing {fp_csv.name}: run scripts/s03_platforms.py first")
        return 2
    if not Path(a.taxonomy_csv).exists():
        print("missing routes_taxonomy.csv")
        return 2
    routes = pd.read_parquet(rr)
    for c in ("rel_path", "copies_rel", "car_fp", "version", "dongle_log", "n_segments"):
        if c not in routes.columns:
            routes[c] = None
    tax = load_taxonomy(a.taxonomy_csv)
    master = load_master(a.route_master)
    audit = load_audit(L.rebase(P.ROUTE_AUDIT, a.data_root))
    fp2plat = S.fp2platform_dict(fp_csv)
    specs = S.read_platform_specs(specs_csv)
    tables = L.taxonomy_tables(a.taxonomy_dir)
    try:
        overrides = read_overrides(L.rebase(P.LABEL_OVERRIDES_ROUTES, a.data_root))
    except ValueError as e:
        print(f"ACCEPTANCE FAILED: {mask(str(e))}")
        return 2

    merges_csv = Path(a.specs_dir) / "model_merges.csv"
    merges = pd.read_csv(merges_csv, dtype=str, keep_default_na=False) if merges_csv.exists() else None
    df, stats = build_labels(routes, tax, master, audit, fp2plat, specs, tables, overrides, merges)
    hours = L.hours_from_segments(routes["n_segments"]).reset_index(drop=True)
    print(f"routes {len(df)}; taxonomy rows {len(tax)}; master rows {len(master)}; audit rows {len(audit)}; "
          f"specs platforms {len(specs)}; overrides {stats['overrides']}; model merges {stats['model_merges']}")
    print("label_source: " + ", ".join(f"{k}={v}" for k, v in df["label_source"].value_counts().items()))
    print("label_grade: " + ", ".join(f"{k}={v}" for k, v in df["label_grade"].value_counts().items()))
    print("powertrain_raw: " + ", ".join(f"{k}={v}" for k, v in df["powertrain_raw"].value_counts(dropna=False).items()))
    print("region: " + ", ".join(f"{k}={v}" for k, v in df["region"].value_counts().items()))
    print("software_era: " + ", ".join(f"{k}={v}" for k, v in df["software_era"].value_counts().items()))
    n_plat = int(df["platform"].notna().sum())
    n_fp = int(df["footprint_m2"].notna().sum())
    h_fp = float(hours[df["footprint_m2"].notna().to_numpy()].sum())
    print(f"platform known: {n_plat}/{len(df)} routes; footprint known: {n_fp} routes, {h_fp:.0f} h "
          f"({100 * h_fp / max(float(hours.sum()), 1e-9):.1f} % of hours); unresolved fingerprints (no mapping row): "
          f"{len(stats['missing_fingerprints'])}; folder spellings without a mapping row: "
          f"{len(stats['missing_folder_spellings'])} ({stats['routes_missing_folder_spelling']} routes)")
    print(cell_table_text(df, hours))
    problems = acceptance(df, routes, stats)
    if problems:
        print("ACCEPTANCE FAILED: " + " | ".join(mask(p) for p in problems))
        return 2
    if a.dry_run:
        print("dry-run: nothing written")
        return 0
    pub = df[PUBLIC_COLUMNS].copy()
    priv = df[PUBLIC_COLUMNS + PRIVATE_EXTRA].copy()
    write_parquet_atomic(pub, L.rebase(P.ROUTES_LABELS, a.data_root))
    write_parquet_atomic(priv, L.rebase(P.LABELS_PRIVATE, a.data_root), private=True)
    print(f"wrote routes_labels.parquet ({len(pub)} rows, {len(PUBLIC_COLUMNS)} cols) and labels_private.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
