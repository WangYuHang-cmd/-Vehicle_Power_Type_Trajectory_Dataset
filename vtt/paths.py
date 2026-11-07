"""Every path of the build, derived from vtt.config.DATA_ROOT. Nothing here touches the raw library."""
from __future__ import annotations

from pathlib import Path

from vtt import config as C


class P:
    DATA = C.DATA_ROOT
    PRIVATE = C.DATA_ROOT / "private"            # chmod 700; never uploaded, never exported
    WORK = C.DATA_ROOT / "work"                  # extractor output (= s02 --data-root): signals/, tables/, logs/
    BUILD = C.DATA_ROOT / "build"
    RELEASE = C.DATA_ROOT / "release" / C.RELEASE_VERSION
    GITHUB_EXPORT = C.DATA_ROOT / "github_export"
    TMP = C.DATA_ROOT / "tmp"
    HF_HOME = C.DATA_ROOT / "hf_home"
    EXT = C.DATA_ROOT / "ext"

    # private
    SEEDS = PRIVATE / "seeds"
    INPUTS = PRIVATE / "inputs"
    PRIVATE_LOGS = PRIVATE / "logs"
    SALT = PRIVATE / "salt.txt"
    RELEASE_SEED = PRIVATE / "release_seed.txt"
    UID_MAP = PRIVATE / "uid_map.parquet"                 # tocf.schema.PRIVATE_UID_MAP (what s02 reads)
    ROUTE_PROBE = PRIVATE / "route_probe.parquet"
    RAW_MANIFEST = PRIVATE / "raw_manifest.csv"
    ROUTES_RESOLVED = PRIVATE / "routes_resolved.parquet"
    IDENTITY_MERGES = PRIVATE / "identity_merges.csv"
    MISFILED_DIRS = PRIVATE / "misfiled_dirs.json"
    INHOUSE_GROUPS = PRIVATE / "inhouse_groups.csv"
    FORBIDDEN_TOKENS = PRIVATE / "forbidden_tokens.txt"
    EXCLUDE_ROUTES = PRIVATE / "exclude_routes.csv"
    ROUTE_AUDIT = PRIVATE / "route_audit.parquet"
    LABELS_PRIVATE = PRIVATE / "labels_private.parquet"
    PSEUDONYM_MAP = PRIVATE / "pseudonym_map.parquet"
    PRIVATE_QC = PRIVATE / "qc"

    # work (extractor)
    SIGNALS = WORK / "signals"
    SEGMENTS = WORK / "tables" / "segments.parquet"
    WORK_LOGS = WORK / "logs"
    S02_FAILURES = WORK / "logs" / "s02_failures.csv"

    # build
    ROUTES_ALL = BUILD / "routes_all.csv"
    ROUTES_SMOKE = BUILD / "routes_smoke20.csv"
    CHUNKS = BUILD / "chunks"
    ROUTES_LABELS = BUILD / "routes_labels.parquet"
    DRIVE_NAMES = BUILD / "drive_names.parquet"
    DATA_DIR = BUILD / "data"                             # data/<vehicle_id>/<drive_id>.parquet
    INDEX = BUILD / "index"
    DRIVES = INDEX / "drives.parquet"
    WINDOWS = INDEX / "windows.parquet"
    HUMAN_ONLY_WINDOWS = INDEX / "human_only_windows.parquet"
    VEHICLES = INDEX / "vehicles.parquet"
    PLATFORMS = INDEX / "platforms.parquet"
    SPLIT_SUMMARY = INDEX / "split_summary.csv"
    SPLITS = BUILD / "splits"
    SPLITS_JSON = SPLITS / "splits.json"
    QC = BUILD / "qc"
    REVIEW_MD = QC / "REVIEW.md"
    LABEL_OVERRIDES = BUILD / "label_overrides.csv"             # S7: scope,key,powertrain,label_note,resolve (drive / vehicle / platform)
    LABEL_OVERRIDES_ROUTES = BUILD / "label_overrides_routes.csv"  # S3a: route_uid,field,value,reason (per-route label / spec cells)
    BUILD_LOGS = BUILD / "logs"
    SIZE_INVENTORY = BUILD / "SIZE_INVENTORY.md"

    # release
    REL_README = RELEASE / "README.md"
    REL_CHECKSUMS = RELEASE / "CHECKSUMS.sha256"
    GATE_OK = RELEASE / ".gate_ok"


def ensure_dirs() -> None:
    for d in (P.WORK, P.BUILD, P.RELEASE, P.GITHUB_EXPORT, P.TMP, P.HF_HOME, P.EXT, P.CHUNKS, P.INDEX, P.SPLITS, P.QC,
              P.BUILD_LOGS, P.SIGNALS, P.WORK_LOGS, P.DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def drive_file(vehicle_id: str, drive_id: str, root: Path | None = None) -> Path:
    return (root or P.DATA_DIR) / vehicle_id / f"{drive_id}.parquet"


def signal_dir(route_uid: str) -> Path:
    return P.SIGNALS / route_uid
