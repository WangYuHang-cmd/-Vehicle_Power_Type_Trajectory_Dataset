"""VTT build configuration: every design switch lives here.

Hard rule: no identifier (dongle / route id, VIN, person or folder name) may ever appear in this file.
Folder re-keys and lab-device lists are loaded at run time from the private root (see vtt.private).
"""
from __future__ import annotations

import os
from pathlib import Path

CONFIG_VERSION = "vtt-1.0"
RELEASE_VERSION = "v1"

# ---------------------------------------------------------------- roots (environment overrides)
DATA_ROOT = Path(os.environ.get("VTT_DATA_ROOT", "$VTT_DATA_ROOT"))
RAW_ROOT = Path(os.environ.get("VTT_RAW_ROOT", "$VTT_RAW_ROOT"))            # read-only raw library
TOCF_ROOT = Path(os.environ.get("VTT_TOCF_ROOT", "$VTT_TOCF_ROOT"))  # extractor + helpers (reused, not vendored)
TAXONOMY_DIR = Path(os.environ.get("VTT_TAXONOMY_DIR", "$VTT_TAXONOMY_DIR"))
ROUTE_MASTER_CSV = Path(os.environ.get("VTT_ROUTE_MASTER", "$VTT_ROUTE_MASTER"))
OPENDBC_ROOT = Path(os.environ.get("VTT_OPENDBC_ROOT", str(DATA_ROOT / "ext" / "opendbc")))   # upstream clone (user)
OPENDBC_LOCAL_FALLBACK = Path(os.environ.get("VTT_OPENDBC_LOCAL", "$VTT_OPENDBC_LOCAL"))  # 2024-08 copy
VENV_PY = os.environ.get("VTT_VENV_PY", "$PY")   # build / tests interpreter
CONDA_PY = os.environ.get("VTT_CONDA_PY", "$CPY")                   # s10_upload + starter kit

# ---------------------------------------------------------------- extraction (ADAS-TO-CF s02, run unchanged)
EXTRACTOR_SCRIPT = TOCF_ROOT / "scripts" / "s02_extract_canonical.py"
EXTRACTOR_VERSION = "s02-v4"          # route_meta.extractor_version expected in work/signals/<uid>/
EXTRACT_WORKERS = 8
EXTRACT_WORKERS_LONG = 4
CHUNK_ROUTES = 400
LONG_ROUTE_SEGMENTS = 150             # routes with more segments go to the final 'long' chunk (4 workers)
# hardlinked into work/signals when the private salt equals the seed salt (env overrides let the tests point both at a synthetic tree)
REUSE_EXISTING_SIGNALS = Path(os.environ.get("VTT_REUSE_EXISTING_SIGNALS", "$VTT_TOCF_DATA/v1/signals"))
SEED_PRIVATE_ROOT = Path(os.environ.get("VTT_SEED_PRIVATE_ROOT", "$VTT_TOCF_DATA/.private"))   # salt.txt + route_probe.parquet seeds

# ---------------------------------------------------------------- labels
POWERTRAIN4 = ("ICE", "HEV", "PHEV", "BEV")
POWERTRAIN_MAP = {"ICE": "ICE", "HEV": "HEV", "PHEV": "PHEV", "BEV": "BEV", "MIXED": None, "UNK": None, "FCEV": None}
MAIN_POWERTRAIN_TASK = ("BEV", "ICE")           # user ruling 2026-11-05: round-1 main experiment is BEV vs ICE
SIZE_CLASSES = ("SMALL", "MID", "LARGE")
# footprint = wheelbase_m * (track_front_m + track_rear_m) / 2  [m^2]; thresholds are locked in S3c
# (docs/SIZE_CLASSES.md). While None, s08_splits refuses to run.
FOOTPRINT_THRESHOLDS_M2: tuple[float, float] | None = (4.43, 4.88)
SIZE_CLASSES_DOC = "docs/SIZE_CLASSES.md"
SIZE_CLASSES_SHA256: str | None = "f87b64e6097c5594ed02a5ce47248a74818c93bb31f484f043f4c8822a14d4d4"
K_ANON_VEHICLES = 1                             # user ruling 2026-11-05: publish all model names (DriveDNA practice)
DROP_UNKNOWN_MODEL = True                       # user ruling: routes of an unknown car model are not released (S9 drops them)
REGION_BUCKETS = {"US": "north_america", "Canada": "north_america", "Europe": "europe", "Asia": "asia"}
REGION_OTHER = "other_unknown"
SOFTWARE_ERAS = ("0.8", "0.9", "0.10", "0.11", "2025", "2026", "other")
LABEL_GRADE_A_SOURCES = ("fingerprint", "alias", "prefix", "fw-match", "resolved_new")

# ---------------------------------------------------------------- per-drive table (sample-indexed, see vtt.schema)
NOMINAL_HZ = 10
NOMINAL_DT_S = 0.1
CS_GAP_S = 0.3              # carState inter-sample dt above this marks a gap row
CTRL_MAX_AGE_S = 0.3        # carControl / controlsState / selfdriveState zero-order-hold max age
LEAD_MAX_AGE_S = 0.75       # radarState (4 Hz) ZOH max age
POSE_MAX_AGE_S = 0.625      # livePose 5 Hz / liveLocationKalman 4 Hz ZOH max age (= tocf POSE_MAX_AGE_S)
POSE_YAW_SIGN = -1.0        # yaw left-positive = -angularVelocityDevice_z
MOVING_V_MPS = 1.0
LEAD_VALID_DREL_MIN_M = 0.5
THW_V_MIN_MPS = 0.5
TTC_CLOSING_MIN_MPS = 0.1
THW_CF_S = 6.0              # car-following when THW <= 6 s (EV-accel run; the paper used 3.5 s)
TORQUE_NORM_Q = 0.95        # steer_torque_norm = raw / quantile(|raw| | moving & steer_pressed)
TORQUE_NORM_MIN_SAMPLES = 50
DEAD_CHANNEL_MIN_PRESSED = 10   # gas/brake magnitude all-zero while *_pressed fires >= this often -> dead channel

# ---------------------------------------------------------------- windows
WINDOW_SAMPLES = 600
WINDOW_SPAN_S = (59.5, 60.5)        # design range of the real span of 600 consecutive samples; public tables carry nominal
                                    # time only, so S6 enforces it as 'no gap row inside the window' (gap = dt > CS_GAP_S)
                                    # and span_s is nominal; the real per-drive span lives in the private QC frame (S5)
WINDOW_MIN_MOVING_FRAC = 0.2
HUMAN_GUARD_S = 10.0                # human_only_strict: no DAS activity in the preceding 10 s either
ACCEL_EP = dict(min_s=3.0, min_dv_mps=2.0, min_mean_a=0.1, eps_mps=0.3, sg_win=5, sg_order=2,
                trim_win_s=0.0)     # 0 = literal EV-paper rule (a monotone run absorbs the steady cruise around a manoeuvre);
                                    # > 0 (e.g. 2.0 s) trims run ends whose central slope is below min_mean_a before the tests
LEAD_ACCEL_EVENT = dict(slope_on=0.5, hold_s=1.0, quiet_s=2.0, quiet_abs=0.3, resp_a=0.3, resp_hold_s=0.5, horizon_s=8.0)

# ---------------------------------------------------------------- splits
SPLIT_SEED = 20261105
SPLIT_FRACS = {"train": 0.70, "validation": 0.15, "test": 0.15}
# user ruling 2026-11-05: the hold-out unit of split_platform is the taxonomy car model (column of the drives index); set to
# "platform" to hold out fingerprint platforms instead
HOLDOUT_KEY = "model"
PLATFORM_HOLDOUT_FRAC = 0.15
PLATFORM_HOLDOUT_MIN_GROUPS = 3
SUPPORTED_CELL = dict(min_eval_groups=4, min_eval_hours_frac=0.24)   # constraints are asserted only in supported cells
EVAL_SHARE_TOL = 0.03
EVAL_GROUP_CAP = 0.50
BALANCED_EVAL = dict(max_group_share=0.25, max_platform_share=0.50)
LOW_SUPPORT = dict(min_test_hours=10.0, min_groups=5, min_platforms=2)
CLASS_WEIGHT_POWER = 0.5
CLASS_WEIGHT_CAP = 10.0
INHOUSE_POLICY = "train_only"       # lab devices never reach validation / test

# ---------------------------------------------------------------- pseudonyms (fresh namespace, random order)
VEHICLE_FMT = "vehicle_{:04d}"
DRIVE_FMT = "drive_{:06d}"
GROUP_FMT = "group_{:04d}"
# identity_merges.csv evidence kinds S4 does NOT merge on (a date-style route id shared across car folders has 1-s
# resolution and may be a coincidence; S4's own route_id rule refuses the same case). Counted, reported, never merged.
MERGE_SKIP_EVIDENCE = ("date_id_other_car",)

# ---------------------------------------------------------------- release
HF_REPO = "HenryYHW/Vehicle_Power_Type_Trajectory_Dataset"
GITHUB_REPO = "WangYuHang-cmd/-Vehicle_Power_Type_Trajectory_Dataset"
DATA_LICENSE = "cc-by-4.0"
CODE_LICENSE = "MIT"
CITATION_ARXIV = "2603.06986"
# columns a model may use under the official protocol (everything else is metadata / provenance)
OFFICIAL_INPUTS = (
    "v_mps", "a_mps2", "steer_angle_deg", "steer_rate_degps", "steer_torque_norm", "yaw_rate_radps", "curv_1pm",
    "lat_acc_mps2", "gas_pressed", "brake_pressed", "steer_pressed", "standstill", "das_enabled",
    "lead_valid", "lead_d_m", "lead_v_rel_mps", "lead_speed_mps", "lead_a_slope_mps2", "thw_s", "ttc_s",
)
