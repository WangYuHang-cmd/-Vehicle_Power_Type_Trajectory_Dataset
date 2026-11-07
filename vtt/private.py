"""Private-root helpers: permissions, salt, folder re-keys, lab devices, forbidden tokens, atomic writers, masking.

Everything under the private root carries real identifiers; it is never uploaded, exported or printed.
"""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path

import pandas as pd

from vtt.paths import P

HEX16 = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{16}(?![0-9a-zA-Z])")
HEX12 = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{12}(?![0-9a-zA-Z])")
ROUTE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}")
ROUTE_NUM = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{8}--[0-9a-f]{10}(?![0-9a-zA-Z])")


def ensure_private_root(root: Path = P.PRIVATE) -> Path:
    """Create the private root with mode 0700 and tighten every file to 0600; refuse to run when it is wider."""
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    for dirpath, dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o700)
        for f in filenames:
            os.chmod(Path(dirpath) / f, 0o600)
    mode = stat.S_IMODE(os.stat(root).st_mode)
    if mode & 0o077:
        raise PermissionError(f"private root is group/world accessible (mode {oct(mode)}): {root}")
    return root


def salt(root: Path = P.PRIVATE) -> str:
    """The internal uid salt. Seeded from seeds/salt.txt on first use (so route_uids match the reused extraction)."""
    p = root / "salt.txt"
    if not p.exists():
        seed = root / "seeds" / "salt.txt"
        if not seed.exists():
            raise FileNotFoundError("no private/salt.txt and no seeds/salt.txt; copy the ADAS-TO-CF salt first")
        p.write_text(seed.read_text())
        os.chmod(p, 0o600)
    s = p.read_text().strip()
    if len(s) < 16:
        raise ValueError("salt too short")
    return s


def misfiled_dirs(root: Path = P.PRIVATE) -> dict[str, str]:
    """folder name -> logged dongle id for dongle folders that are not the device's own id (person names, wrong ids).
    Union of the private JSON and (when importable) the extractor's own table."""
    out: dict[str, str] = {}
    try:
        from vtt.tocf_bridge import tocf
        out.update(dict(tocf("config").MISFILED_DONGLE_DIRS))
    except Exception:
        pass
    p = root / "misfiled_dirs.json"
    if p.exists():
        out.update(json.loads(p.read_text()))
    return out


def forbidden_tokens(root: Path = P.PRIVATE) -> list[str]:
    p = root / "forbidden_tokens.txt"
    if not p.exists():
        return []
    return [t.strip() for t in p.read_text().splitlines() if t.strip() and not t.startswith("#")]


def inhouse_groups(root: Path = P.PRIVATE) -> pd.DataFrame:
    """Lab-owned devices (train-only). Columns: dongle_log, source."""
    p = root / "inhouse_groups.csv"
    if not p.exists():
        return pd.DataFrame(columns=["dongle_log", "source"])
    return pd.read_csv(p, dtype=str)


def _tighten(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def write_parquet_atomic(df: pd.DataFrame, path: Path, private: bool = False, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False, **kw)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if private:
        _tighten(path)
    return path


def write_json_atomic(obj, path: Path, private: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)
    if private:
        _tighten(path)
    return path


def write_text_atomic(text: str, path: Path, private: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    if private:
        _tighten(path)
    return path


def mask(text: str) -> str:
    """Replace identifier-like tokens before anything is printed or logged."""
    text = ROUTE_DATE.sub("<route_date>", str(text))
    text = ROUTE_NUM.sub("<route_num>", text)
    text = HEX16.sub("<hex16>", text)
    return text
