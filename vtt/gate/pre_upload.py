"""Pre-upload gate: scan a release tree, write the masked report and the `.gate_ok` marker (0 findings only).

`.gate_ok` holds {utc, n_files, checksums_sha256} so that the upload stage can verify it refers to the tree it ships.
Exit code: 0 iff no findings, else 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vtt import config as C
from vtt.gate import scan as SC
from vtt.gate.rules import Report


def run(release_root: Path, data_root: Path = C.DATA_ROOT, private_root: Path | None = None, workers: int = 4,
        sample_rows: int = 50000, write_ok: bool = True) -> tuple[Report, Path]:
    """Scan, write the report, maintain `.gate_ok`; returns (report, report_path)."""
    release_root = Path(release_root)
    private_root = Path(private_root) if private_root else Path(data_root) / "private"
    rep = SC.scan_release(release_root, private_root=private_root, workers=workers, sample_rows=sample_rows)
    report_path = SC.write_report(rep, data_root=Path(data_root), private_root=private_root)
    ok_path = release_root / ".gate_ok"
    if rep.ok and write_ok:
        ck = release_root / "CHECKSUMS.sha256"
        marker = {"utc": rep.utc, "n_files": rep.n_files, "checksums_sha256": SC.sha256_file(ck) if ck.exists() else None}
        ok_path.write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n")
    elif not rep.ok and ok_path.exists():
        ok_path.unlink()
    return rep, report_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VTT pre-upload gate (exit 0 iff no findings)")
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--release-root", type=Path, default=None, help="default <data-root>/release/<version>")
    ap.add_argument("--private-root", type=Path, default=None, help="default <data-root>/private")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample-rows", type=int, default=50000,
                    help="rows of each drive table / window shard whose string values are checked (index and splits tables: all)")
    ap.add_argument("--no-mark", action="store_true", help="do not write .gate_ok on success")
    a = ap.parse_args(argv)
    release_root = a.release_root or (a.data_root / "release" / C.RELEASE_VERSION)
    rep, path = run(release_root, data_root=a.data_root, private_root=a.private_root, workers=a.workers,
                    sample_rows=a.sample_rows, write_ok=not a.no_mark)
    SC.print_summary(rep, path)
    return 0 if rep.ok else 2


if __name__ == "__main__":
    sys.exit(main())
