#!/usr/bin/env python3
"""S9b: run the release gates on their own.

  pre-upload (default): scan <release-root> (default <data-root>/release/<version>), write the masked report and
                        .gate_ok on 0 findings.
  --pre-push REPO:      run the code-repository gate on a git checkout (GitHub export).
Exit 0 iff no findings, else 2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vtt import config as C
from vtt.gate import pre_push, pre_upload, scan as SC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--release-root", type=Path, default=None)
    ap.add_argument("--private-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample-rows", type=int, default=50000,
                    help="rows of each drive table / window shard whose string values are checked (index and splits tables: all)")
    ap.add_argument("--no-mark", action="store_true", help="do not write .gate_ok on success")
    ap.add_argument("--pre-push", type=Path, default=None, metavar="REPO", help="run the code-repository gate instead")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the `import vtt.config` probe (pre-push)")
    a = ap.parse_args(argv)
    private_root = a.private_root or (a.data_root / "private")
    if a.pre_push is not None:
        rep = pre_push.run(a.pre_push, private_root=private_root, python=a.python)
        path = SC.write_report(rep, data_root=a.data_root, private_root=private_root)
        SC.print_summary(rep, path)
        return 0 if rep.ok else 2
    release_root = a.release_root or (a.data_root / "release" / C.RELEASE_VERSION)
    rep, path = pre_upload.run(release_root, data_root=a.data_root, private_root=private_root, workers=a.workers,
                               sample_rows=a.sample_rows, write_ok=not a.no_mark)
    SC.print_summary(rep, path)
    return 0 if rep.ok else 2


if __name__ == "__main__":
    sys.exit(main())
