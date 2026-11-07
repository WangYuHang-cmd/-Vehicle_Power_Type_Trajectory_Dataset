"""Verify and complete the hardlink / copy reuse of an earlier extraction (s02_extract_all.sh step 1).

Every reused route directory under the destination must be a file-for-file image of its source directory: same file names
and same sizes (a copy cut by a killed cp or by EXDEV on a merged file system is caught by the size). Destination directories
whose name is also a source route are told apart as
  complete   - image of the source (counted as hardlinked when every file shares the source's inode),
  fresh      - route_meta.json newer than the source's: extracted by this build, never removed, never compared,
  incomplete - anything else: removed with --fix so that the extractor (which only looks for route_meta.json) does not trust it.
--fill rebuilds every source route (one with a route_meta.json) that has no complete / fresh counterpart, file by file
(os.link, shutil.copy2 when the link fails) into a temporary '.<uid>.reusing' directory renamed into place, so a destination
directory is either whole or absent. Exit 0 only when every source route is complete or fresh in the destination
('missing' = 0). Counts only, no ids.

  $PY -m vtt.reuse --src <REUSE_EXISTING_SIGNALS> --dst <work/signals> [--fix] [--fill]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

META = "route_meta.json"
RESULT_KEYS = ("source_dirs", "source_usable", "dest_dirs", "complete", "fresh", "incomplete", "removed", "filled", "filled_linked_files",
               "filled_copied_files", "fill_failed", "missing", "hardlinked")


def _files(d: Path) -> dict[str, int]:
    """{name: size} of the regular files directly under d (symlinks are not followed)."""
    return {e.name: e.stat(follow_symlinks=False).st_size for e in os.scandir(d) if e.is_file(follow_symlinks=False)}


def _inodes(d: Path) -> dict[str, int]:
    return {e.name: e.stat(follow_symlinks=False).st_ino for e in os.scandir(d) if e.is_file(follow_symlinks=False)}


def _meta_mtime(d: Path) -> float | None:
    try:
        return os.stat(d / META).st_mtime
    except OSError:
        return None


def _route_dirs(root: Path) -> set[str]:
    return {e.name for e in os.scandir(root) if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")}


def _rmtree(d: Path) -> bool:
    """Remove a destination directory we own, re-opening its mode first when a cp -a preserved an unreadable one."""
    try:
        shutil.rmtree(d)
        return True
    except OSError:
        try:
            os.chmod(d, 0o700)
            shutil.rmtree(d)
            return True
        except OSError:
            return False


def fill_one(s: Path, d: Path) -> tuple[int, int]:
    """Rebuild d as an image of s: every file linked (else copied with metadata) into a temporary sibling that is renamed into
    place, so d is either whole or absent. Returns (files linked, files copied); raises OSError when s cannot be read or d
    cannot be created (an existing d is never touched)."""
    tmp = d.parent / f".{d.name}.reusing"
    if tmp.exists() and not _rmtree(tmp):
        raise OSError(f"cannot clear the temporary directory of a previous fill")
    tmp.mkdir()
    n_link = n_copy = 0
    try:
        for name in sorted(_files(s)):
            try:
                os.link(s / name, tmp / name)
                n_link += 1
            except OSError:
                shutil.copy2(s / name, tmp / name)
                n_copy += 1
        os.rename(tmp, d)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return n_link, n_copy


def verify(src: Path, dst: Path, fix: bool = False, fill: bool = False) -> dict:
    """Compare dst against src (RESULT_KEYS). source_usable = source routes with a route_meta.json (the others cannot be reused
    and are ignored); missing = usable source routes without a complete or fresh counterpart after --fix / --fill."""
    src, dst = Path(src), Path(dst)
    out = {k: 0 for k in RESULT_KEYS}
    if not src.is_dir():
        return out
    src_dirs = _route_dirs(src)
    usable: set[str] = set()
    for n in src_dirs:
        try:
            has_meta = (src / n / META).exists()
        except OSError:                                    # unreadable source dir: cannot be ruled out, so it must be reused later
            has_meta = True
        if has_meta:
            usable.add(n)
    out["source_dirs"], out["source_usable"] = len(src_dirs), len(usable)
    satisfied: set[str] = set()
    if dst.is_dir():
        for name in sorted(_route_dirs(dst)):
            out["dest_dirs"] += 1
            if name not in usable:
                continue                                   # extracted fresh by this build (or its source is unusable): not a reused dir
            s, d = src / name, dst / name
            sm, dm = _meta_mtime(s), _meta_mtime(d)
            if dm is not None and sm is not None and dm > sm:
                out["fresh"] += 1                          # a reused dir can never be newer than its source
                satisfied.add(name)
                continue
            try:
                same = _files(s) == _files(d)
            except OSError:
                same = False
            if same:
                out["complete"] += 1
                satisfied.add(name)
                try:
                    if _inodes(s) == _inodes(d):
                        out["hardlinked"] += 1
                except OSError:
                    pass
            else:
                out["incomplete"] += 1
                if fix and _rmtree(d):
                    out["removed"] += 1
    missing = sorted(usable - satisfied)
    if fill and missing:
        dst.mkdir(parents=True, exist_ok=True)
        for name in missing:
            if (dst / name).exists():
                out["fill_failed"] += 1                    # incomplete and kept (no --fix): cannot rebuild over it
                continue
            try:
                n_link, n_copy = fill_one(src / name, dst / name)
            except OSError:
                out["fill_failed"] += 1
                continue
            out["filled"] += 1
            out["filled_linked_files"] += n_link
            out["filled_copied_files"] += n_copy
            satisfied.add(name)
            if n_copy == 0:
                out["hardlinked"] += 1
    out["missing"] = len(usable - satisfied)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--fix", action="store_true", help="remove incomplete reused directories")
    ap.add_argument("--fill", action="store_true", help="rebuild missing source routes (link, else copy; atomic per directory)")
    a = ap.parse_args(argv)
    res = verify(Path(a.src), Path(a.dst), fix=a.fix, fill=a.fill)
    print("reuse check: " + " ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    return 0 if res["missing"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
