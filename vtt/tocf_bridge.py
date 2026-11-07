"""Access to the ADAS-TO-CF extractor and its helpers without vendoring them.

The package lives at config.TOCF_ROOT (not part of this repository). Only pure, identifier-free helpers are used:
raw-library scanning / probing (s00, s01), extraction (s02, run as a subprocess), zero-order hold, lead and pose helpers.
"""
from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

from vtt import config as C


def _ensure_path() -> None:
    root = str(C.TOCF_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def tocf(mod: str):
    """Import tocf.<mod> (e.g. 'rawlib', 'extract', 'engage', 'lead', 'pose', 'tables', 'capnp_io', 'schema')."""
    _ensure_path()
    return importlib.import_module(f"tocf.{mod}")


def load_script(name: str):
    """Import a tocf scripts/<name>.py as a module (same pattern as the tocf test-suite)."""
    _ensure_path()
    path = C.TOCF_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # multiprocessing pickles worker functions by module name
    spec.loader.exec_module(mod)
    return mod


def extractor_available() -> bool:
    return C.EXTRACTOR_SCRIPT.exists()


def extractor_cmd(routes_csv: Path, data_root: Path, private_root: Path, workers: int = C.EXTRACT_WORKERS,
                  raw_root: Path = C.RAW_ROOT, frame_count: bool = False, dry_run: bool = False,
                  python: str = C.VENV_PY, nice: int | None = 15, extra: tuple[str, ...] = ()) -> list[str]:
    """Command line that runs the extractor unchanged with VTT roots."""
    cmd: list[str] = []
    if nice is not None:
        cmd += ["nice", "-n", str(nice), "ionice", "-c3"]
    cmd += [python, str(C.EXTRACTOR_SCRIPT), "--routes", str(routes_csv), "--raw-root", str(raw_root),
            "--data-root", str(data_root), "--private-root", str(private_root), "--workers", str(workers)]
    if not frame_count:
        cmd.append("--no-frame-count")
    if dry_run:
        cmd.append("--dry-run")
    cmd += list(extra)
    return cmd


def run_extractor(routes_csv: Path, data_root: Path, private_root: Path, **kw) -> int:
    cmd = extractor_cmd(routes_csv, data_root, private_root, **kw)
    return subprocess.call(cmd, cwd=str(C.TOCF_ROOT))
