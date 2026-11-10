#!/usr/bin/env python3
"""Platform folding and labels for the commaCarSegments (CCS) adapter.

Shared by ccs_select.py (the owner's download list) and ccs_decode.py (rlog -> windows). It imports no opendbc: the
MIGRATION table and the platform name list are read with the AST parsers of vtt.specs from an opendbc source tree
(default: the PyPI opendbc installed in <data-root>/ext/venv_opendbc, fallback: the 2024-08 openpilot
checkout), so the module runs under $CPY as well as under the decode venv.

Label sources, in this order:
  1. the lab platform table vtt/specs_data/platform_specs.csv (footprint = wheelbase x mean track), powertrain from
     build/index/drives.parquet (per platform, as used in training) or the taxonomy DIRS table;
  2. experiments/transfer/specs_external.csv (public specs for the CCS keys the lab table lacks), joined on platform_keys;
  3. nothing (footprint NaN, powertrain unknown) - the platform can still be selected and decoded when opendbc knows it.
Stratum per car model (split_platform scheme of the drives index): 'seen' = the model has training / validation drives,
'held-out' = the model is one of the 41 held-out test models, 'absent' = the model is not in the lab inventory (or only
in excluded drives). No device or route identifier is read or written here.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vtt import config as C  # noqa: E402
from vtt import specs as S  # noqa: E402

DATA_ROOT = Path(C.DATA_ROOT)
EXT_ROOT = Path(os.environ.get("VTT_EXTERNAL", str(DATA_ROOT / "external")))
CCS_DIR = EXT_ROOT / "ccs"
VENV_DIR = DATA_ROOT / "ext" / "venv_opendbc"
LAB_SPECS = REPO / "vtt" / "specs_data" / "platform_specs.csv"
EXT_SPECS = REPO / "experiments" / "transfer" / "specs_external.csv"
DRIVES = DATA_ROOT / "build" / "index" / "drives.parquet"
TAXONOMY = Path(C.TAXONOMY_DIR) / "taxonomy_tables.py"
THRESHOLDS = tuple(C.FOOTPRINT_THRESHOLDS_M2)
BORDERLINE_M2 = 0.05
DROP_KEYS = {"MOCK"}                           # never a car
POWERTRAIN_VOCAB = {"ICE": "ICE", "HEV": "HEV", "PHEV": "PHEV", "BEV": "BEV", "FCEV": "FCEV"}   # MIXED / UNK -> unknown
STRATA = ("seen", "held-out", "absent")


# ----------------------------------------------------------------------------- opendbc source tree (AST only)
def find_opendbc_root(explicit: str | Path | None = None) -> Path:
    """First directory that holds opendbc/car/fingerprints.py among: the argument, $VTT_OPENDBC_PY_ROOT, the decode venv's
    site-packages, vtt.config.OPENDBC_ROOT, the 2024-08 openpilot checkout."""
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("VTT_OPENDBC_PY_ROOT"):
        cands.append(Path(os.environ["VTT_OPENDBC_PY_ROOT"]))
    cands += [Path(p) for p in sorted(glob.glob(str(VENV_DIR / "lib" / "python3*" / "site-packages")))]
    cands += [Path(C.OPENDBC_ROOT), Path(C.OPENDBC_LOCAL_FALLBACK)]
    for c in cands:
        if (c / "opendbc" / "car" / "fingerprints.py").is_file():
            return c
    raise FileNotFoundError("no opendbc source tree found (tried: " + ", ".join(str(c) for c in cands) + ")")


def opendbc_version(root: Path) -> str:
    """Version string of the opendbc tree (dist-info of a pip install, else the directory name)."""
    for d in sorted(root.glob("opendbc-*.dist-info")):
        return d.name.replace("opendbc-", "").replace(".dist-info", "") + " (pip)"
    return root.name


def load_opendbc_names(root: Path) -> tuple[dict[str, str], set[str]]:
    """(MIGRATION legacy key -> platform, set of platform names) parsed from the opendbc tree without importing it."""
    migration = S.parse_migration(root)
    platforms = set(S.parse_platforms(root)["platform"].astype(str))
    return migration, platforms


def fold(key: str, migration: dict[str, str]) -> str:
    """The current opendbc platform name of an index key (legacy spellings folded with MIGRATION)."""
    return migration.get(key, key)


# ----------------------------------------------------------------------------- pseudonyms
def pseudonym(kind: str, *parts: str) -> str:
    """Stable pseudonym of a device / route (sha1 prefix); kind is 'v' (vehicle) or 'd' (drive)."""
    h = hashlib.sha1(("ccs|" + "|".join(parts)).encode()).hexdigest()[:10]
    return f"ccs_{kind}_{h}"


# ----------------------------------------------------------------------------- labels
def footprint_from_mm(wb, tf, tr) -> float:
    try:
        wb, tf, tr = float(wb), float(tf), float(tr)
    except (TypeError, ValueError):
        return float("nan")
    if not (wb > 0 and tf > 0 and tr > 0):
        return float("nan")
    return wb / 1000.0 * (tf + tr) / 2.0 / 1000.0


def size_class(fp: float) -> str | None:
    if fp is None or not math.isfinite(fp):
        return None
    lo, hi = THRESHOLDS
    return "SMALL" if fp < lo else ("MID" if fp < hi else "LARGE")


def is_borderline(fp: float) -> bool:
    return bool(fp is not None and math.isfinite(fp) and min(abs(fp - THRESHOLDS[0]), abs(fp - THRESHOLDS[1])) < BORDERLINE_M2)


def _taxonomy_dirs(path: Path) -> dict:
    """DIRS table of the hand-curated taxonomy module (platform -> (model, powertrain, note)); {} when absent."""
    if not path.is_file():
        return {}
    spec = importlib.util.spec_from_file_location("ccs_taxonomy_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(getattr(mod, "DIRS", {}))


class Labeller:
    """platform (folded) -> model, footprint, size class, powertrain, stratum; see the module docstring."""

    def __init__(self, lab_specs: Path = LAB_SPECS, ext_specs: Path = EXT_SPECS, drives: Path = DRIVES, taxonomy: Path = TAXONOMY):
        lab = pd.read_csv(lab_specs, dtype=str, keep_default_na=False)
        self.lab: dict[str, dict] = {}
        for r in lab.itertuples(index=False):
            self.lab[r.platform] = dict(model=r.model or None, brand=r.brand or None, body=r.body or None,
                                        footprint_m2=footprint_from_mm(r.wheelbase_mm, r.track_front_mm, r.track_rear_mm),
                                        wheelbase_mm=r.wheelbase_mm, confidence=r.spec_confidence)
        self.drive_pt: dict[str, str] = {}
        self.drive_model: dict[str, str] = {}
        self.drive_fp: dict[str, float] = {}
        self.seen: set[str] = set()
        self.heldout: set[str] = set()
        self.excluded_only: set[str] = set()
        if Path(drives).is_file():
            d = pd.read_parquet(drives, columns=["platform", "model", "powertrain_raw", "footprint_m2", "split_platform"])
            d = d[d["platform"].notna()]
            for p, g in d.groupby("platform"):
                self.drive_pt[p] = str(g["powertrain_raw"].mode().iloc[0]) if g["powertrain_raw"].notna().any() else "UNK"
                self.drive_model[p] = str(g["model"].mode().iloc[0])
                fp = g["footprint_m2"].dropna()
                self.drive_fp[p] = float(fp.iloc[0]) if len(fp) else float("nan")
            m = d[d["model"].notna() & (d["model"] != "Unknown")]
            splits = m.groupby("model")["split_platform"].agg(lambda s: set(s.dropna()))
            self.seen = {k for k, v in splits.items() if v & {"train", "validation"}}
            self.heldout = {k for k, v in splits.items() if "test" in v}
            self.excluded_only = {k for k, v in splits.items() if v and v <= {"excluded"}}
        self.dirs = _taxonomy_dirs(Path(taxonomy))
        self.ext: dict[str, dict] = {}
        if Path(ext_specs).is_file():
            e = pd.read_csv(ext_specs, dtype=str, keep_default_na=False)
            for r in e.itertuples(index=False):
                for k in [x.strip() for x in r.platform_keys.replace(";", " ").replace(",", " ").split() if x.strip()]:
                    self.ext.setdefault(k, r._asdict())

    # -- pieces
    def _powertrain_raw(self, *keys: str) -> str:
        for k in keys:
            if k in self.drive_pt and self.drive_pt[k] not in ("UNK", "", "None"):
                return self.drive_pt[k]
        for k in keys:
            if k in self.dirs and len(self.dirs[k]) >= 2 and self.dirs[k][1]:
                return str(self.dirs[k][1])
        for k in keys:
            if k in self.ext and self.ext[k].get("powertrain"):
                p = self.ext[k]["powertrain"].upper()
                return p if p in POWERTRAIN_VOCAB else "UNK"
        return "UNK"

    def stratum(self, model: str | None) -> str:
        if model and model in self.seen:
            return "seen"
        if model and model in self.heldout:
            return "held-out"
        return "absent"

    def label(self, folded: str, raw_key: str | None = None) -> dict:
        """Label dict of a platform; `raw_key` (the index spelling) is tried after the folded name."""
        keys = [k for k in (folded, raw_key) if k]
        out = dict(platform=folded, model=None, brand=None, footprint_m2=float("nan"), size_class=None, powertrain="unknown",
                   powertrain_raw="UNK", borderline=False, label_source="none", stratum="absent", note="")
        row = next((self.lab[k] for k in keys if k in self.lab), None)
        if row is not None:
            out.update(model=row["model"], brand=row["brand"], footprint_m2=row["footprint_m2"], label_source="lab")
            if not math.isfinite(out["footprint_m2"]):
                fp = next((self.drive_fp[k] for k in keys if k in self.drive_fp and math.isfinite(self.drive_fp[k])), float("nan"))
                out["footprint_m2"] = fp
            if not out["model"]:
                out["model"] = next((self.drive_model[k] for k in keys if k in self.drive_model), None)
        else:
            ext = next((self.ext[k] for k in keys if k in self.ext), None)
            if ext is not None:
                fp = footprint_from_mm(ext.get("wheelbase_mm"), ext.get("track_front_mm"), ext.get("track_rear_mm"))
                if not math.isfinite(fp):
                    try:
                        fp = float(ext.get("footprint_m2") or "nan")
                    except ValueError:
                        fp = float("nan")
                out.update(model=ext.get("model") or None, footprint_m2=fp, label_source="public",
                           note=f"public spec ({ext.get('confidence', '')} confidence)")
            elif any(k in self.drive_model for k in keys):
                k = next(k for k in keys if k in self.drive_model)
                out.update(model=self.drive_model[k], footprint_m2=self.drive_fp.get(k, float("nan")), label_source="lab-drives")
            elif any(k in self.dirs for k in keys):
                k = next(k for k in keys if k in self.dirs)
                out.update(model=str(self.dirs[k][0]), label_source="taxonomy")
        pt_raw = self._powertrain_raw(*keys)
        out["powertrain_raw"] = pt_raw
        out["powertrain"] = POWERTRAIN_VOCAB.get(pt_raw, "unknown")
        if pt_raw in ("MIXED",):
            out["note"] = (out["note"] + "; " if out["note"] else "") + "platform merges gasoline and electrified variants"
        out["size_class"] = size_class(out["footprint_m2"])
        out["borderline"] = is_borderline(out["footprint_m2"])
        out["stratum"] = self.stratum(out["model"])
        if out["model"] in self.excluded_only:
            out["note"] = (out["note"] + "; " if out["note"] else "") + "model only in excluded lab drives (never trained on)"
        return out


__all__ = ["REPO", "DATA_ROOT", "EXT_ROOT", "CCS_DIR", "VENV_DIR", "LAB_SPECS", "EXT_SPECS", "DRIVES", "TAXONOMY", "THRESHOLDS",
           "DROP_KEYS", "STRATA", "find_opendbc_root", "opendbc_version", "load_opendbc_names", "fold", "pseudonym",
           "footprint_from_mm", "size_class", "is_borderline", "Labeller"]
