"""Loading the release from the Hugging Face Hub or from a local folder (release tree, build tree or `sample/`).

Remote access uses `huggingface_hub.hf_hub_download` (gated dataset: run `hf auth login` first); every file is
cached under HF_HOME. Local access reads the same relative paths from `root`. Nothing here needs torch.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

try:
    from vtt import config as C
    DEFAULT_REPO, OFFICIAL_INPUTS = C.HF_REPO, tuple(C.OFFICIAL_INPUTS)
except ImportError:                                      # pragma: no cover - mirrors vtt.config
    DEFAULT_REPO = "HenryYHW/Vehicle_Power_Type_Trajectory_Dataset"
    OFFICIAL_INPUTS = ("v_mps", "a_mps2", "steer_angle_deg", "steer_rate_degps", "steer_torque_norm", "yaw_rate_radps", "curv_1pm",
                       "lat_acc_mps2", "gas_pressed", "brake_pressed", "steer_pressed", "standstill", "das_enabled",
                       "lead_valid", "lead_d_m", "lead_v_rel_mps", "lead_speed_mps", "lead_a_slope_mps2", "thw_s", "ttc_s")

INDEX_FILES = {                                          # candidates in order (release name first, build name second)
    "drives": ("index/drives.parquet",),
    "windows": ("index/windows_60s.parquet", "index/windows.parquet"),
    "vehicles": ("index/vehicles.parquet",),
    "platforms": ("index/platforms.parquet",),
    "split_summary": ("index/split_summary.csv",),
    "splits": ("splits/splits.json",),
}
# S8 file naming of the balanced evaluation lists: splits/eval_balanced_<task>_<split><suffix>.parquet per split scheme
BALANCED_SUFFIX = {"split_platform": "", "split_device": "_device"}


def fetch(rel_path: str, root: Path | str | None = None, repo: str = DEFAULT_REPO, revision: str | None = None) -> Path:
    """Local path of one release file: `root/rel_path` when root is given, else downloaded from the Hub."""
    if root is not None:
        p = Path(root) / rel_path
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p
    from huggingface_hub import hf_hub_download           # noqa: WPS433 - optional dependency for remote use
    return Path(hf_hub_download(repo, rel_path, repo_type="dataset", revision=revision))


def _read(path: Path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"unsupported file type: {path.suffix}")


def load_index(name: str, root: Path | str | None = None, repo: str = DEFAULT_REPO):
    """One index table (`drives`, `windows`, `vehicles`, `platforms`, `split_summary`) or the `splits` dict."""
    errors = []
    for rel in INDEX_FILES[name]:
        try:
            return _read(fetch(rel, root, repo))
        except Exception as exc:                          # noqa: BLE001 - try the next candidate name
            errors.append(f"{rel}: {type(exc).__name__}")
    raise FileNotFoundError(f"no {name} index found ({'; '.join(errors)})")


def balanced_list_path(task: str, split: str = "test", split_col: str = "split_platform") -> str:
    """Relative path of the balanced evaluation list for (task, split) under one split scheme: the headline
    `split_platform` lists carry no suffix, the `split_device` lists end in `_device`."""
    if split_col not in BALANCED_SUFFIX:
        raise ValueError(f"unknown split column {split_col!r}; choose from {tuple(BALANCED_SUFFIX)}")
    return f"splits/eval_balanced_{task}_{split}{BALANCED_SUFFIX[split_col]}.parquet"


def load_balanced_ids(task: str, split: str = "test", root: Path | str | None = None, repo: str = DEFAULT_REPO,
                      split_col: str = "split_platform") -> set[str]:
    """Window ids of the published balanced evaluation list for (task, split) of the given split scheme."""
    df = _read(fetch(balanced_list_path(task, split, split_col), root, repo))
    return set(df["window_id"].astype(str))


def load_drive(ref: str, root: Path | str | None = None, repo: str = DEFAULT_REPO, drives: pd.DataFrame | None = None) -> pd.DataFrame:
    """One drive table by relative file path (`data/<vehicle_id>/<drive_id>.parquet`) or by drive_id (needs the
    drives index, loaded when not given)."""
    rel = ref
    if not ref.endswith(".parquet"):
        if drives is None:
            drives = load_index("drives", root, repo)
        hit = drives.loc[drives["drive_id"] == ref, "file"]
        if hit.empty:
            raise KeyError(f"drive_id not in index: {ref}")
        rel = str(hit.iloc[0])
    return pd.read_parquet(fetch(rel, root, repo))


def window_tensor(drive: pd.DataFrame, row0: int, row1: int, channels=OFFICIAL_INPUTS, dtype=np.float32) -> np.ndarray:
    """(row1 - row0, len(channels)) array of the drive rows [row0, row1); booleans become 0/1, NaN is kept
    (undefined values such as thw_s without a lead are the caller's to handle); missing channels are NaN."""
    n = int(row1) - int(row0)
    out = np.full((n, len(channels)), np.nan, dtype=dtype)
    for j, ch in enumerate(channels):
        if ch not in drive:
            warnings.warn(f"channel {ch} missing from the drive table")
            continue
        col = drive[ch].iloc[int(row0):int(row1)]
        out[:, j] = col.to_numpy(dtype=float) if col.dtype != object else pd.to_numeric(col, errors="coerce").to_numpy(float)
    return out


def iter_windows(split: str = "test", protocol: str = "test_full", task: str = "bev_ice", root: Path | str | None = None,
                 repo: str = DEFAULT_REPO, split_col: str = "split_platform", windows: pd.DataFrame | None = None,
                 drives: pd.DataFrame | None = None, channels=OFFICIAL_INPUTS, limit: int | None = None) -> Iterator[tuple[pd.Series, np.ndarray]]:
    """Yield (window index row, tensor) for the windows a protocol selects, reading every drive table once.
    For the train / validation splits `protocol='test_full'` simply means 'every eligible window of the task';
    `test_balanced` reads the published list of the chosen split scheme (`split_col`, see balanced_list_path)."""
    from starter_kit.eval import protocol_mask
    windows = load_index("windows", root, repo) if windows is None else windows
    drives = load_index("drives", root, repo) if drives is None else drives
    ids = load_balanced_ids(task, split, root, repo, split_col) if protocol == "test_balanced" else None
    sub = windows[protocol_mask(windows, task, protocol, split_col, split, ids)].sort_values(["drive_id", "w_idx"])
    if limit is not None:
        sub = sub.head(limit)
    for drive_id, grp in sub.groupby("drive_id", sort=False):
        drive = load_drive(str(drive_id), root, repo, drives)
        for _, row in grp.iterrows():
            yield row, window_tensor(drive, int(row["row0"]), int(row["row1"]), channels)
