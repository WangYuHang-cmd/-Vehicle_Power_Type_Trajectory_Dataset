#!/usr/bin/env python3
"""S9: assemble release/<version> from build/, render the dataset card, write CHECKSUMS and run the pre-upload gate.

Exit 0 when the release is assembled and the gate reports no finding (`.gate_ok` written), 2 otherwise: a failed
acceptance check, a gate finding, or any error while assembling (missing drive table, template / numbers mismatch,
unreadable parquet) is reported as one masked line, the traceback only with --debug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vtt import config as C
from vtt.package import AcceptanceError, Roots, log, package


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--build-root", type=Path, default=None, help="default <data-root>/build")
    ap.add_argument("--release-root", type=Path, default=None, help="default <data-root>/release/<version>")
    ap.add_argument("--private-root", type=Path, default=None, help="default <data-root>/private")
    ap.add_argument("--split-col", default="split_platform", choices=["split_platform", "split_device"],
                    help="split column that assigns windows to train / validation / test shards")
    ap.add_argument("--shard-windows", type=int, default=5000, help="windows per shard file (<= 500 MB each)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--copy", action="store_true", help="copy drive tables instead of hardlinking them")
    ap.add_argument("--force", action="store_true", help="rebuild existing shards")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--allow-no-splits", action="store_true", help="package even when no window has a split (smoke tests)")
    ap.add_argument("--contact-email", default=None, help="contact address on the card (default: gate_allow.json)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true", help="re-raise errors with their traceback (default: one masked line, exit 2)")
    a = ap.parse_args(argv)
    roots = Roots.from_data_root(a.data_root, a.build_root, a.release_root, a.private_root)
    try:
        summary = package(roots, split_col=a.split_col, per_shard=a.shard_windows, workers=a.workers, copy=a.copy, force=a.force,
                          dry_run=a.dry_run, run_gate=not a.skip_gate, allow_no_splits=a.allow_no_splits, contact_email=a.contact_email)
    except AcceptanceError as e:
        log(f"s09: ACCEPTANCE FAILED: {e}")
        return 2
    except Exception as e:                            # noqa: BLE001 - any assembly error is a failed stage, never a traceback
        if a.debug:
            raise
        log(f"s09: FAILED ({type(e).__name__}): {e}")
        return 2
    log(f"s09: accepted={summary['accepted']} gate_findings={summary.get('gate_findings', 'skipped')}")
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
