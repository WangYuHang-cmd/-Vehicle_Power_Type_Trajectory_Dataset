"""Route labels: fingerprint -> model / powertrain (port of the taxonomy build's resolve()), ECU rules, grades, buckets,
footprint and size class.

Units: wheelbase / track mm, footprint m^2 (= wheelbase_m x mean track_m), hours h.
The hand-curated tables (MODELS, DIRS, ALIAS) are imported from config.TAXONOMY_DIR at run time (read-only); tests may
inject their own tables object. No identifier is read or written here.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from vtt import config as C
from vtt.paths import P

UNKNOWN_MODEL = ("Unknown", "unknown", "UNK")     # brand, body, size_code when the model is not in MODELS
_VER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")

# (regex on the normalized fingerprint, model, powertrain) fallbacks for the prefix rule, checked in order
GENERIC = [
    (r"PLUG-IN_HYBRID|PLUGIN|_PHEV", None, "PHEV"),
    (r"HYBRID|_HEV|_H$|_H_", None, "HEV"),
    (r"_EV\b|_EV_|ELECTRIC|LIGHTNING|MACH|BOLT|LEAF|IONIQ_5|IONIQ_6|IONIQ_9|EV6|EV9|TESLA|RIVIAN|POLESTAR|ENYAQ|BORN|ID4|ID_4|ATTO|MARVEL|SOLTERRA",
     None, "BEV"),
]
HONDA_ACCORD_ECU_PLATFORMS = ("HONDA_ACCORD", "HONDA_ACCORD_2018", "HONDA_INSPIRE")

# known label noise: (regex on the normalized platform / fingerprint spelling, note)
LABEL_NOISE = (
    (r"^TOYOTA_PRIUS(_TSS2|_2017)?$", "platform shared with Prius Prime (PHEV); labelled HEV"),
    (r"^RIVIAN_R1(_|$)", "R1T pickup and R1S SUV share the platform; body and footprint differ"),
    (r"^HONDA_PILOT", "platform shared with Honda Passport"),
    (r"^RAM_1500", "eTorque 48 V mild hybrid counted as ICE"),
    (r"^FORD_FOCUS_(MK4|4TH_GEN)", "48 V mild-hybrid variants counted as ICE"),
    (r"^LANDROVER_DEFENDER", "mild-hybrid variants counted as ICE"),
    (r"^CHEVROLET_BOLT_EUV", "platform also covers Bolt EV 2022-23 (hatch)"),
    (r"^SUBARU_IMPREZA", "platform also covers Crosstrek"),
    (r"^CHEVROLET_SILVERADO", "platform also covers GMC Sierra 1500"),
    (r"^HYUNDAI_PALISADE", "platform also covers Kia Telluride"),
)


# ------------------------------------------------------------------------------------------------- taxonomy tables
_TABLES: dict[str, object] = {}


def taxonomy_tables(taxonomy_dir: Path | None = None):
    """The hand-curated taxonomy module (MODELS, DIRS, ALIAS) imported from TAXONOMY_DIR; cached."""
    d = Path(taxonomy_dir or C.TAXONOMY_DIR)
    key = str(d)
    if key not in _TABLES:
        path = d / "taxonomy_tables.py"
        if not path.exists():
            raise FileNotFoundError(f"taxonomy_tables.py not found under the taxonomy dir")
        spec = importlib.util.spec_from_file_location(f"taxonomy_tables_{abs(hash(key))}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # the module only defines dicts (its __main__ block is not executed)
        _TABLES[key] = SimpleNamespace(MODELS=dict(mod.MODELS), DIRS=dict(mod.DIRS), ALIAS=dict(mod.ALIAS))
    return _TABLES[key]


def make_tables(models: dict, dirs: dict, alias: dict):
    """Tables object for tests / injection (same shape as taxonomy_tables())."""
    return SimpleNamespace(MODELS=dict(models), DIRS=dict(dirs), ALIAS=dict(alias))


def normalize(fp) -> str:
    return re.sub(r"\s+", "_", str(fp).strip()).upper()


def _dirs_norm(tables) -> dict[str, str]:
    cache = getattr(tables, "_DIRS_N", None)
    if cache is None:
        cache = {normalize(k): k for k in tables.DIRS}
        try:
            tables._DIRS_N = cache
        except AttributeError:
            pass
    return cache


# ------------------------------------------------------------------------------------------------- resolve (port)
def resolve(fp, tables=None):
    """fingerprint string -> (model, powertrain, how) with how in fingerprint | alias | prefix | unmapped;
    None for empty / mock fingerprints. Faithful port of the taxonomy build (exact DIRS -> ALIAS -> prefix stripping
    with GENERIC powertrain overrides)."""
    T = tables or taxonomy_tables()
    if not isinstance(fp, str) or not fp.strip() or fp.strip().lower() == "mock":
        return None
    n = normalize(fp)
    dn = _dirs_norm(T)
    if n in dn:
        k = dn[n]
        return (T.DIRS[k][0], T.DIRS[k][1], "fingerprint")
    if n in T.ALIAS:
        return (T.ALIAS[n][0], T.ALIAS[n][1], "alias")
    toks = n.split("_")
    while len(toks) > 1:
        toks = toks[:-1]
        c = "_".join(toks)
        if c in dn:
            k = dn[c]
            mdl, pt, _ = T.DIRS[k]
            for rx, _, ptx in GENERIC:
                if re.search(rx, n) and not re.search(rx, c):
                    pt = ptx
                    break
            return (mdl, pt, "prefix")
    return ("?" + fp, "UNK", "unmapped")


def dir_note(fp, tables=None) -> str:
    """The curator's note of the DIRS entry matching the fingerprint exactly ('' otherwise)."""
    T = tables or taxonomy_tables()
    if not isinstance(fp, str):
        return ""
    k = _dirs_norm(T).get(normalize(fp))
    if k is None:
        return ""
    v = T.DIRS[k]
    return str(v[2]) if len(v) > 2 and v[2] else ""


def model_attrs(model, tables=None) -> tuple[str, str, str]:
    """(brand, body, size_code) of a canonical model; unknown models fall back to the first word as brand."""
    T = tables or taxonomy_tables()
    if not isinstance(model, str) or not model or model.startswith("?"):
        return UNKNOWN_MODEL
    return T.MODELS.get(model, (model.split(" ")[0], "unknown", "UNK"))


def resolve_full(fp, tables=None) -> dict:
    """Everything the labels need from a fingerprint: model, brand, body, size_code, powertrain_raw, how, note.
    Unmapped or empty fingerprints give the Unknown model with how = unmapped / none."""
    T = tables or taxonomy_tables()
    res = resolve(fp, T)
    if res is None:
        return dict(model="Unknown", brand="Unknown", body="unknown", size_code="UNK", powertrain_raw="UNK", how="none", note="")
    mdl, pt, how = res
    if how == "unmapped":
        return dict(model="Unknown", brand="Unknown", body="unknown", size_code="UNK", powertrain_raw="UNK", how="unmapped",
                    note="fingerprint not in the taxonomy tables")
    brand, body, size = model_attrs(mdl, T)
    return dict(model=mdl, brand=brand, body=body, size_code=size, powertrain_raw=pt, how=how, note=dir_note(fp, T))


# ------------------------------------------------------------------------------------------------------- ECU rules
def ecu_set(ecus) -> set[str]:
    """ECU names from a JSON list string, a ';'-joined string, or any iterable; empty set for missing values."""
    if ecus is None or (isinstance(ecus, float) and math.isnan(ecus)):
        return set()
    if isinstance(ecus, str):
        s = ecus.strip()
        if not s:
            return set()
        if s.startswith("["):
            try:
                return {str(x) for x in json.loads(s)}
            except ValueError:
                return set()
        return {t.strip() for t in s.split(";") if t.strip()}
    try:
        return {str(x) for x in ecus}
    except TypeError:
        return set()


def ecu_resolve(model, fp, powertrain_raw, ecus=None, transmission=None) -> tuple[str, str]:
    """Route-level powertrain for MIXED platforms from the car's ECU list / transmission type.

    Returns (powertrain_raw, rule) where rule is '' when nothing applied. Rules (as in the taxonomy build):
    Toyota / Lexus: hybrid ECU -> HEV, else transmission ECU -> ICE; Honda Accord 2018-22 platforms: transmission or
    programmedFuelInjection ECU -> ICE, shiftByWire + eps without them -> HEV; Volkswagen Golf: transmission 'direct' -> BEV."""
    if powertrain_raw != "MIXED" or not isinstance(model, str):
        return powertrain_raw, ""
    ec = ecu_set(ecus)
    if model.startswith("Toyota") or model.startswith("Lexus"):
        if "hybrid" in ec:
            return "HEV", "toyota_hybrid_ecu"
        if "transmission" in ec:
            return "ICE", "toyota_transmission_ecu"
    fpn = normalize(fp) if isinstance(fp, str) else ""
    if model == "Honda Accord" and fpn in HONDA_ACCORD_ECU_PLATFORMS and ec:
        if "transmission" in ec or "programmedFuelInjection" in ec:
            return "ICE", "honda_engine_ecu"
        if "shiftByWire" in ec and "eps" in ec:
            return "HEV", "honda_hybrid_ecu_pattern"
    if model == "Volkswagen Golf" and str(transmission or "").strip().lower() == "direct":
        return "BEV", "vw_direct_transmission"
    return powertrain_raw, ""


# ------------------------------------------------------------------------------------------- model -> platform fallback
def model_platform_index(specs) -> dict[str, list[str]]:
    """model -> sorted platforms of platform_specs whose `model` cell is that model (empty models skipped)."""
    out: dict[str, set[str]] = {}
    if specs is None or not len(specs) or "model" not in specs.columns:
        return {}
    for plat, mdl in zip(specs["platform"], specs["model"]):
        if isinstance(mdl, str) and mdl.strip() and isinstance(plat, str) and plat.strip():
            out.setdefault(mdl.strip(), set()).add(plat.strip())
    return {k: sorted(v) for k, v in out.items()}


def platform_from_model(model, powertrain_raw, index: dict[str, list[str]], tables=None) -> tuple[str | None, int]:
    """(platform, n_candidates) for a route whose fingerprint is mock / unreadable but whose model is known (folder
    name or firmware match). The platform is the unique platform_specs row carrying the model; when several do, the
    candidates whose own powertrain (resolve(platform name)) contradicts the route's powertrain_raw are dropped first.
    (None, n) when none or several remain."""
    if not isinstance(model, str) or not model.strip() or model.strip() == "Unknown" or model.startswith("?"):
        return None, 0
    cands = list(index.get(model.strip(), []))
    if len(cands) > 1 and isinstance(powertrain_raw, str) and powertrain_raw in C.POWERTRAIN4:
        keep = []
        for p in cands:
            res = resolve(p, tables)
            pt = res[1] if res is not None and res[2] != "unmapped" else None
            if pt is None or pt == "MIXED" or pt == powertrain_raw:
                keep.append(p)
        cands = keep
    if len(cands) == 1:
        return cands[0], 1
    return None, len(cands)


# ---------------------------------------------------------------------------------------------- codes and buckets
def powertrain_map(powertrain_raw) -> str | None:
    """7-value platform code -> 4-class label or None (MIXED / UNK / FCEV / unknown)."""
    if not isinstance(powertrain_raw, str):
        return None
    return C.POWERTRAIN_MAP.get(powertrain_raw.strip().upper())


def label_grade(label_source) -> str:
    """A: identified from the route's own fingerprint; B: from the folder name only; C: unknown."""
    if isinstance(label_source, str) and label_source in C.LABEL_GRADE_A_SOURCES:
        return "A"
    if label_source == "directory":
        return "B"
    return "C"


def region_bucket(region) -> str:
    if not isinstance(region, str):
        return C.REGION_OTHER
    return C.REGION_BUCKETS.get(region.strip(), C.REGION_OTHER)


def version_tuple(version) -> tuple[int, int, int] | None:
    m = _VER_RE.search(str(version or ""))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


def software_era(version) -> str:
    """Coarse software generation from the logger's version string: 0.8 / 0.9 / 0.10 / 0.11 / 2025 / 2026 / other.
    Defers to vtt.manifest.software_era when that module exists (S1 writes the same value into routes_all.csv)."""
    try:
        from vtt.manifest import software_era as _era   # type: ignore
        era = _era(version)
        if era in C.SOFTWARE_ERAS:
            return era
    except Exception:            # noqa: BLE001  module absent or incompatible: use the local rule
        pass
    vt = version_tuple(version)
    if vt is None:
        return "other"
    major, minor = vt[0], vt[1]
    if major == 0:
        era = f"0.{minor}"
        return era if era in C.SOFTWARE_ERAS else "other"
    if str(major) in C.SOFTWARE_ERAS:
        return str(major)
    return "other"


def label_note(platform=None, fingerprint=None, extra: str | None = None) -> str | None:
    """Known label-noise note for a platform / fingerprint spelling, joined with `extra` (curator or rule notes)."""
    parts: list[str] = []
    for key in (platform, fingerprint):
        if not isinstance(key, str) or not key:
            continue
        n = normalize(key)
        for rx, note in LABEL_NOISE:
            if re.search(rx, n) and note not in parts:
                parts.append(note)
        if parts:
            break
    if isinstance(extra, str) and extra.strip():
        for p in extra.split("; "):
            if p.strip() and p.strip() not in parts:
                parts.append(p.strip())
    return "; ".join(parts) if parts else None


# ----------------------------------------------------------------------------------------- footprint and size class
def footprint_m2(wheelbase_mm, track_front_mm, track_rear_mm) -> float:
    """wheelbase_m x (track_front_m + track_rear_m) / 2; NaN when any input is missing."""
    vals = [wheelbase_mm, track_front_mm, track_rear_mm]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return float("nan")
    try:
        wb, tf, tr = (float(v) for v in vals)
    except (TypeError, ValueError):
        return float("nan")
    if wb <= 0 or tf <= 0 or tr <= 0:
        return float("nan")
    return wb / 1000.0 * (tf + tr) / 2000.0


def footprint_series(df: pd.DataFrame) -> pd.Series:
    """Vectorised footprint: the table's footprint_m2 where filled, else computed from wheelbase and tracks."""
    wb = pd.to_numeric(df.get("wheelbase_mm"), errors="coerce")
    tf = pd.to_numeric(df.get("track_front_mm"), errors="coerce")
    tr = pd.to_numeric(df.get("track_rear_mm"), errors="coerce")
    calc = wb / 1000.0 * (tf + tr) / 2000.0
    calc = calc.where((wb > 0) & (tf > 0) & (tr > 0))
    if "footprint_m2" in df.columns:
        given = pd.to_numeric(df["footprint_m2"], errors="coerce")
        return given.where(given.notna(), calc)
    return calc


def size_class(footprint, thresholds: tuple[float, float] | None = None) -> str | None:
    """SMALL (< t1) / MID ([t1, t2)) / LARGE (>= t2) by footprint; None while thresholds are not locked
    (config.FOOTPRINT_THRESHOLDS_M2 is None and none given) or the footprint is unknown."""
    thr = C.FOOTPRINT_THRESHOLDS_M2 if thresholds is None else thresholds
    if thr is None or footprint is None:
        return None
    try:
        f = float(footprint)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    t1, t2 = float(thr[0]), float(thr[1])
    if t2 < t1:
        t1, t2 = t2, t1
    if f < t1:
        return C.SIZE_CLASSES[0]
    if f < t2:
        return C.SIZE_CLASSES[1]
    return C.SIZE_CLASSES[2]


def cut_size_classes(footprints, thresholds: tuple[float, float] | None = None) -> pd.Series:
    """Vectorised size_class (object series with None for unknown)."""
    s = pd.Series(footprints)
    thr = C.FOOTPRINT_THRESHOLDS_M2 if thresholds is None else thresholds
    out = pd.Series([None] * len(s), index=s.index, dtype=object)
    if thr is None:
        return out
    f = pd.to_numeric(s, errors="coerce")
    t1, t2 = sorted((float(thr[0]), float(thr[1])))
    out[f < t1] = C.SIZE_CLASSES[0]
    out[(f >= t1) & (f < t2)] = C.SIZE_CLASSES[1]
    out[f >= t2] = C.SIZE_CLASSES[2]
    return out


# ------------------------------------------------------------------------------------------------------ path helper
def rebase(path: Path, data_root: Path | None) -> Path:
    """A vtt.paths.P path re-rooted under another data root (scripts' --data-root); unchanged when root is None."""
    if data_root is None or Path(data_root) == P.DATA:
        return Path(path)
    return Path(data_root) / Path(path).relative_to(P.DATA)


def hours_from_segments(n_segments) -> pd.Series:
    """Approximate hours from 1-min segment counts."""
    return pd.to_numeric(pd.Series(n_segments), errors="coerce").fillna(0) / 60.0


def parse_copies(copies_rel) -> list[str]:
    """copies_rel JSON list -> list of rel_paths (empty on missing / malformed)."""
    if isinstance(copies_rel, (list, tuple, np.ndarray)):
        return [str(x) for x in copies_rel]
    if not isinstance(copies_rel, str) or not copies_rel.strip():
        return []
    try:
        v = json.loads(copies_rel)
    except ValueError:
        return []
    return [str(x) for x in v] if isinstance(v, list) else []


# ------------------------------------------------------------------------------------------------- model merges
def apply_model_merges(df: pd.DataFrame, merges: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Rename taxonomy models listed in vtt/specs_data/model_merges.csv (from_model, to_model, reason): the same
    vehicle platform family recorded under two model names would otherwise be held out separately (and exclude every
    device that drove both). Only `model` changes (platform, specs, body stay); the reason is appended to label_note.
    Returns (frame, {from_model: n_routes})."""
    counts: dict[str, int] = {}
    if merges is None or not len(merges) or "model" not in df.columns:
        return df, counts
    df = df.copy()
    for m in merges.itertuples(index=False):
        src, dst = str(m.from_model).strip(), str(m.to_model).strip()
        if not src or not dst or src == dst:
            continue
        sel = (df["model"] == src).to_numpy()
        n = int(sel.sum())
        if not n:
            continue
        reason = getattr(m, "reason", "")
        note = f"model merged into {dst}" + (f" ({reason.strip()})" if isinstance(reason, str) and reason.strip() else "")
        df.loc[sel, "model"] = dst
        prev = df.loc[sel, "label_note"]
        df.loc[sel, "label_note"] = [f"{p}; {note}" if isinstance(p, str) and p else note for p in prev]
        counts[src] = n
    return df, counts
