#!/usr/bin/env python3
"""S7: QC + label audit -> build/qc/drives_qc.csv, build/qc/REVIEW.md, private/qc/label_audit_private.csv and the
label_note write-back into build/index/drives.parquet (see vtt.qc).

Inputs: build/index/drives.parquet (required), build/index/windows.parquet (optional), private/route_audit.parquet +
build/drive_names.parquet (label audit, optional), private/qc/drives_private_qc.parquet (rate, optional),
private/pseudonym_map.parquet (device x platform, optional), build/label_overrides.csv (optional).
Exit 2 when unresolved label conflicts remain (unless --allow-conflicts) or the write-back fails the schema check.
Prints counts only. --dry-run computes everything in memory and writes nothing (not even the work/qc_parts cache).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from vtt import config as C
from vtt import qc as Q
from vtt import schema as S
from vtt.private import write_text_atomic
from vtt.windows import Roots


def _read_opt(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rebuild the cached per-vehicle scan parts")
    ap.add_argument("--allow-conflicts", action="store_true", help="exit 0 even with unresolved label conflicts")
    ap.add_argument("--dry-run", action="store_true", help="compute and print counts, write nothing")
    a = ap.parse_args(argv)
    roots = Roots(a.data_root)
    if not roots.drives.exists():
        print(f"missing drives index under {roots.index}; run S5 first")
        return 1
    drives = pd.read_parquet(roots.drives)
    windows = _read_opt(roots.windows)
    private_qc = _read_opt(roots.private_drives_qc)
    route_audit = _read_opt(roots.route_audit)
    names = _read_opt(roots.drive_names)
    if names is None:
        names = _read_opt(roots.pseudonym_map)
    pseudonym_map = _read_opt(roots.pseudonym_map)
    overrides = Q.load_overrides(roots.label_overrides)

    scan = Q.scan_all(drives, roots, workers=max(1, a.workers), force=a.force, dry_run=a.dry_run)
    rates = Q.rate_table(drives, private_qc)
    dead = Q.dead_channel_summary(drives)
    lead_flag = Q.lead_source_flags(drives, scan)
    lead_sources = Q.lead_source_table(drives, scan)
    dup = Q.duplicate_of(scan)
    n_multi, multi, basis = Q.multi_platform(drives, pseudonym_map)
    drives2, notes, changed, resolved = Q.apply_overrides(drives, overrides)
    audit = Q.label_audit(drives2, route_audit, names)
    verdicts = audit.set_index("drive_id")["verdict"]
    unresolved = Q.unresolved_conflicts(verdicts, resolved)
    drives3 = Q.write_back_labels(drives2, verdicts, notes, resolved)
    qc_table = Q.drives_qc_table(drives3, scan, rates, lead_flag, dup, multi, audit, resolved, windows)
    review = Q.review_markdown(drives3, windows, qc_table, dead, audit, resolved, n_multi, basis, private_qc is not None,
                               route_audit is not None and names is not None, lead_sources)
    vc = audit["verdict"].value_counts()
    print(f"drives={len(drives)} windows={0 if windows is None else len(windows)} overrides={len(overrides)} "
          f"powertrain_changed={len(changed)} resolved={len(resolved)}")
    print("label_audit: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    print(f"dead_share: gas={dead[0]['gas']:.3f} brake={dead[0]['brake']:.3f} torque={dead[0]['torque']:.3f} "
          f"(expected gas~{Q.EXPECTED_DEAD_SHARE['gas']:.2f} brake~{Q.EXPECTED_DEAD_SHARE['brake']:.2f})")
    print(f"flags: rate={int((qc_table['rate_flag'] != '').sum())} lead={int((qc_table['lead_flag'] != '').sum())} "
          f"duplicates={int((qc_table['duplicate_of'] != '').sum())} yaw_all_zero={int(qc_table['yaw_all_zero'].eq(True).sum())} "
          f"multi_platform_{basis}s={n_multi}")
    print("lead_source by radar_unavailable: " + "; ".join(
        f"unavailable={bool(r.radar_unavailable)} drives={r.drives} radar_share={r.radar_share:.3f} camera_share={r.camera_share:.3f}"
        for r in lead_sources.itertuples()))
    print(f"unresolved_conflicts={len(unresolved)}")
    if a.dry_run:
        print("dry-run: nothing written")
        return 2 if (unresolved and not a.allow_conflicts) else 0

    roots.qc.mkdir(parents=True, exist_ok=True)
    tmp = roots.drives_qc.with_name(".drives_qc.csv.tmp")
    qc_table.to_csv(tmp, index=False)
    os.replace(tmp, roots.drives_qc)
    write_text_atomic(review, roots.review_md)
    if roots.private.exists():
        roots.private_qc.mkdir(parents=True, exist_ok=True)
        os.chmod(roots.private_qc, 0o700)
        write_text_atomic(audit.to_csv(index=False), roots.label_audit_private, private=True)
    S.write_public_parquet(drives3, roots.drives, S.DRIVES_INDEX_COLUMNS, Q.table_name(roots.drives, "drives"))
    n_prop = Q.propagate_to_windows(roots, drives3, changed)
    problems = S.check_public_parquet(roots.drives, S.DRIVES_INDEX_COLUMNS, Q.table_name(roots.drives, "drives"))
    print(f"wrote drives_qc rows={len(qc_table)} review_bytes={len(review)} label_note_updated={len(drives3)} windows_relabelled={n_prop}")
    if problems:
        print("ACCEPTANCE FAILED: " + "; ".join(problems))
        return 2
    if unresolved and not a.allow_conflicts:
        print("ACCEPTANCE FAILED: unresolved label conflicts (see build/qc/REVIEW.md; use label_overrides.csv or --allow-conflicts)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
