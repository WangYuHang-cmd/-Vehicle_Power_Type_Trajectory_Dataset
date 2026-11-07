"""Re-root a vtt.paths.P path onto another data root (scripts take --data-root; P is bound to config.DATA_ROOT at import)."""
from __future__ import annotations

from pathlib import Path

from vtt.paths import P


def reroot(path: Path, data_root: Path | str | None) -> Path:
    """P.X expressed under data_root (unchanged when data_root is None or equals P.DATA)."""
    if data_root is None:
        return Path(path)
    data_root = Path(data_root)
    if data_root == P.DATA:
        return Path(path)
    return data_root / Path(path).relative_to(P.DATA)
