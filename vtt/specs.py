"""Platform specifications: opendbc platform enums (CarSpecs + CarDocs), fingerprint -> platform mapping, specs skeleton.

Units: curb mass kg, wheelbase mm (opendbc stores metres), track mm, footprint m^2.

The opendbc checkout is parsed with `ast` (never imported): `opendbc/car/<brand>/values.py` `class CAR(Platforms)` members
and `opendbc/car/fingerprints.py` MIGRATION. Nothing here reads the raw library or any identifier.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C

SPECS_DIR = Path(__file__).resolve().parent / "specs_data"      # data dir; "specs" would shadow this module
FP2PLATFORM_CSV = SPECS_DIR / "fingerprint_to_platform.csv"
PLATFORM_SPECS_CSV = SPECS_DIR / "platform_specs.csv"

FP2PLATFORM_COLUMNS = ("fingerprint", "platform", "method", "note")
PLATFORM_SPECS_COLUMNS = ("platform", "brand", "model", "body", "generation_note", "wheelbase_mm", "track_front_mm",
                          "track_rear_mm", "footprint_m2", "footprint_min_m2", "footprint_max_m2", "curb_mass_kg",
                          "mass_source", "wheelbase_source", "track_source", "spec_confidence", "source_urls", "note")
SPEC_NUMERIC = ("wheelbase_mm", "track_front_mm", "track_rear_mm", "footprint_m2", "footprint_min_m2", "footprint_max_m2",
                "curb_mass_kg")
METHODS = ("exact", "migration", "suffix", "manual")
UNRESOLVED = "UNRESOLVED"
OPENDBC_SOURCE = "opendbc"

# names that may appear inside CarSpecs(mass=..., wheelbase=...) expressions
_EVAL_ENV = {"CV.LB_TO_KG": 0.453592, "LB_TO_KG": 0.453592, "STD_CARGO_KG": 136.0, "CV.KPH_TO_MS": 1.0 / 3.6,
             "CV.MPH_TO_MS": 0.44704, "CV.MS_TO_KPH": 3.6, "CV.MS_TO_MPH": 1.0 / 0.44704}
# fork / variant tags that do not change the vehicle (stripped one at a time, right to left)
FORK_SUFFIXES = ("MMR", "PEDAL", "ASCM", "NON_SCC", "NO_RADAR", "CAMERA", "NON_ACC", "ACC", "CC", "SCC", "NO_ACC")
_FORK_RE = re.compile(r"_(%s)$" % "|".join(FORK_SUFFIXES))
_YEAR_RE = re.compile(r"_((?:19|20)\d{2})(?:_((?:19|20)\d{2}))?$")
_POWERTRAIN_TOKENS = ("HEV", "PHEV", "EV", "HYBRID", "ELECTRIC", "PLUG", "PRIME", "ELECTRIFIED", "H")
_DOC_YEARS_RE = re.compile(r"((?:19|20)\d{2})(?:-(\d{2,4}))?")


# ---------------------------------------------------------------------------------------------------- opendbc parsing
def opendbc_root() -> Path:
    """config.OPENDBC_ROOT when it holds opendbc/car, else the local 2024-08 fallback copy."""
    if (C.OPENDBC_ROOT / "opendbc" / "car").is_dir():
        return C.OPENDBC_ROOT
    return C.OPENDBC_LOCAL_FALLBACK


def _eval(node, env: dict[str, float] = _EVAL_ENV) -> float | None:
    """Evaluate a numeric expression made of literals, + - * /, unary minus and names from `env`; None otherwise."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, env)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b if b else None
        return None
    if isinstance(node, (ast.Name, ast.Attribute)):
        return env.get(ast.unparse(node))
    return None


def _func_name(call: ast.Call) -> str:
    f = call.func
    return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")


def _carspecs_values(call: ast.Call) -> dict[str, float | None]:
    """mass_kg / wheelbase_mm / steer_ratio from a CarSpecs(...) call (keywords or the first three positionals)."""
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    pos = list(call.args)
    for i, name in enumerate(("mass", "wheelbase", "steerRatio")):
        if name not in kw and i < len(pos):
            kw[name] = pos[i]
    mass = _eval(kw["mass"]) if "mass" in kw else None
    wb = _eval(kw["wheelbase"]) if "wheelbase" in kw else None
    sr = _eval(kw["steerRatio"]) if "steerRatio" in kw else None
    return {"mass_kg": mass, "wheelbase_mm": None if wb is None else wb * 1000.0, "steer_ratio": sr}


def _doc_strings(call: ast.Call) -> list[str]:
    """Model-year strings of every *CarDocs("...") inside a platform config call."""
    out: list[str] = []
    for node in ast.walk(call):
        if isinstance(node, ast.Call) and _func_name(node).endswith("CarDocs") and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out.append(a.value)
    return out


def doc_years(doc: str) -> set[int]:
    """Model years named by a CarDocs string: 'Toyota RAV4 2019-21' -> {2019, 2020, 2021}; 'comma body' -> set()."""
    years: set[int] = set()
    for m in _DOC_YEARS_RE.finditer(doc):
        y0 = int(m.group(1))
        y1 = m.group(2)
        if y1 is None:
            years.add(y0)
            continue
        y1i = int(y1) if len(y1) == 4 else int(str(y0)[:2] + y1)
        if y1i < y0:
            y1i = y0
        years.update(range(y0, min(y1i, y0 + 15) + 1))
    return years


def _spec_source(call: ast.Call, members: dict[str, dict], module_specs: dict[str, ast.Call]) -> dict | None:
    """Specs of a platform config call: inline CarSpecs(...), specs=..., OTHER.specs, or OTHER.override(...)."""
    nodes = list(call.args) + [k.value for k in call.keywords if k.arg in (None, "specs")]
    for n in nodes:
        if isinstance(n, ast.Call) and _func_name(n).endswith("CarSpecs"):
            return _carspecs_values(n)
        if isinstance(n, ast.Attribute) and n.attr == "specs" and isinstance(n.value, ast.Name) and n.value.id in members:
            return dict(members[n.value.id]["specs"])
        if isinstance(n, ast.Name) and n.id in module_specs:
            return _carspecs_values(module_specs[n.id])
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in members:   # X.override(...)
        return dict(members[f.value.id]["specs"])
    return None


def parse_values_file(path: Path) -> dict[str, dict]:
    """platform -> {brand_dir, specs: {mass_kg, wheelbase_mm, steer_ratio}, docs: [str]} for one values.py."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_specs: dict[str, ast.Call] = {}
    for st in tree.body:
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) and _func_name(st.value).endswith("CarSpecs"):
            for t in st.targets:
                if isinstance(t, ast.Name):
                    module_specs[t.id] = st.value
    members: dict[str, dict] = {}
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CAR"
                and any("Platforms" in ast.unparse(b) for b in cls_bases(n))]:
        for st in cls.body:
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                continue
            name = st.targets[0].id
            empty = {"mass_kg": None, "wheelbase_mm": None, "steer_ratio": None}
            docs: list[str] = []
            specs = None
            if isinstance(st.value, ast.Call):
                specs = _spec_source(st.value, members, module_specs)
                docs = _doc_strings(st.value)
                f = st.value.func      # X.override(car_docs=[]) inherits the docs too when none are given
                if not docs and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in members:
                    docs = list(members[f.value.id]["docs"])
            members[name] = {"brand_dir": path.parent.name, "specs": specs or empty, "docs": docs}
    return members


def cls_bases(cls: ast.ClassDef) -> list[ast.expr]:
    return list(cls.bases)


def parse_platforms(root: Path | None = None) -> pd.DataFrame:
    """Every platform of an opendbc checkout. Columns: platform, brand_dir, mass_kg, wheelbase_mm, steer_ratio, docs
    (' | '-joined CarDocs strings), years (sorted model years as 'y0-y1' or '')."""
    root = Path(root or opendbc_root())
    rows = []
    for vp in sorted(root.glob("opendbc/car/*/values.py")):
        for name, m in parse_values_file(vp).items():
            yrs = set()
            for d in m["docs"]:
                yrs |= doc_years(d)
            rows.append({"platform": name, "brand_dir": m["brand_dir"], **m["specs"], "docs": " | ".join(m["docs"]),
                         "years": f"{min(yrs)}-{max(yrs)}" if yrs else ""})
    df = pd.DataFrame(rows, columns=["platform", "brand_dir", "mass_kg", "wheelbase_mm", "steer_ratio", "docs", "years"])
    return df.drop_duplicates("platform").sort_values("platform").reset_index(drop=True)


def parse_migration(root: Path | None = None) -> dict[str, str]:
    """fingerprints.py MIGRATION: legacy fingerprint string -> platform name (BRAND.PLATFORM attribute -> 'PLATFORM')."""
    root = Path(root or opendbc_root())
    p = root / "opendbc" / "car" / "fingerprints.py"
    if not p.exists():
        return {}
    tree = ast.parse(p.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MIGRATION" for t in node.targets)):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                continue
            if isinstance(v, ast.Attribute):
                out[k.value] = v.attr
            elif isinstance(v, ast.Name):
                out[k.value] = v.id
            elif isinstance(v, ast.Constant) and isinstance(v.value, str):
                out[k.value] = v.value
    return out


# ---------------------------------------------------------------------------------------------- fingerprint mapping
def normalize(fp: str) -> str:
    """Fingerprint spelling key: whitespace -> '_', upper case (same rule as the taxonomy build)."""
    return re.sub(r"\s+", "_", str(fp).strip()).upper()


def _doc_years_of(platforms: pd.DataFrame) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for p, docs in zip(platforms["platform"], platforms["docs"]):
        yrs: set[int] = set()
        for d in str(docs or "").split(" | "):
            yrs |= doc_years(d)
        out[p] = yrs
    return out


def _year_candidates(base: str, years: list[int], fp_tokens: set[str], platform_years: dict[str, set[int]]) -> list[str]:
    """Platforms named `base` or `base_*` whose CarDocs cover one of `years` and add no powertrain token the
    fingerprint lacks (so HYUNDAI_ELANTRA_2022 never lands on HYUNDAI_ELANTRA_HEV_2021)."""
    out = []
    for p, yrs in platform_years.items():
        if p != base and not p.startswith(base + "_"):
            continue
        if not (yrs & set(years)):
            continue
        extra = set(p[len(base):].strip("_").split("_")) - {""}
        if any(tok in _POWERTRAIN_TOKENS and tok not in fp_tokens for tok in extra):
            continue
        out.append(p)
    return sorted(out)


def map_fingerprint(fp: str, platforms: pd.DataFrame | set[str], migration: dict[str, str]) -> tuple[str, str, str]:
    """(platform, method, note) for one fingerprint spelling.

    exact: the spelling (raw or normalized) is a platform; migration: the spelling (raw or normalized) is a legacy name in
    MIGRATION; suffix: a fork tag (FORK_SUFFIXES) and/or a trailing model year was stripped and the remainder matched
    exactly, by migration, or by the year-aware docs rule; manual: unresolved, platform = normalized spelling."""
    if isinstance(platforms, pd.DataFrame):
        names = set(platforms["platform"])
        pyears = _doc_years_of(platforms)
    else:
        names = set(platforms)
        pyears = {p: set() for p in names}
    mig_norm = {normalize(k): v for k, v in migration.items()}
    raw = str(fp).strip()
    n = normalize(raw)
    if raw in names:
        return raw, "exact", ""
    if n in names:
        return n, "exact", "normalized spelling"
    if raw in migration and migration[raw] in names:
        return migration[raw], "migration", ""
    if n in mig_norm and mig_norm[n] in names:
        return mig_norm[n], "migration", "normalized spelling"
    cand = n
    stripped: list[str] = []
    for _ in range(4):                                     # fork tags, right to left
        new = _FORK_RE.sub("", cand)
        if new == cand or "_" not in new:
            break
        stripped.append(cand[len(new):])
        cand = new
        if cand in names:
            return cand, "suffix", f"stripped {''.join(reversed(stripped))}"
        if cand in mig_norm and mig_norm[cand] in names:
            return mig_norm[cand], "suffix", f"stripped {''.join(reversed(stripped))}; migration"
    m = _YEAR_RE.search(cand)
    if m:
        years = [int(y) for y in m.groups() if y]
        base = cand[: m.start()]
        fp_tokens = set(base.split("_"))
        cands = _year_candidates(base, years, fp_tokens, pyears)
        if len(cands) == 1:
            return cands[0], "suffix", f"year {'/'.join(map(str, years))} matched CarDocs of {cands[0]}"
        if len(cands) > 1 and base in cands:
            return base, "suffix", f"year {'/'.join(map(str, years))}; several candidates, kept base ({'|'.join(cands)})"
        if cands:
            return n, "manual", f"{UNRESOLVED}; year candidates: {'|'.join(cands)}"
    return n, "manual", UNRESOLVED


def map_fingerprints(fps: list[str], platforms: pd.DataFrame, migration: dict[str, str],
                     existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """fingerprint_to_platform table for all spellings (sorted). Rows of `existing` that were resolved by hand
    (method manual, note not starting with UNRESOLVED) are kept verbatim."""
    keep: dict[str, dict] = {}
    if existing is not None and len(existing):
        for r in existing.to_dict("records"):
            note = str(r.get("note") or "")
            if str(r.get("method")) == "manual" and note and not note.startswith(UNRESOLVED):
                keep[str(r["fingerprint"])] = {k: ("" if pd.isna(r.get(k)) else r.get(k)) for k in FP2PLATFORM_COLUMNS}
    rows = []
    for fp in sorted({str(f).strip() for f in fps if isinstance(f, str) and str(f).strip()}):
        if fp in keep:
            rows.append(keep[fp])
            continue
        plat, method, note = map_fingerprint(fp, platforms, migration)
        rows.append({"fingerprint": fp, "platform": plat, "method": method, "note": note})
    return pd.DataFrame(rows, columns=list(FP2PLATFORM_COLUMNS))


def read_fp2platform(path: Path = FP2PLATFORM_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=list(FP2PLATFORM_COLUMNS))
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in FP2PLATFORM_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[list(FP2PLATFORM_COLUMNS)]


def fp2platform_dict(path: Path = FP2PLATFORM_CSV) -> dict[str, str]:
    """fingerprint spelling (raw, as logged) -> platform; also keyed by the normalized spelling and by platform names."""
    df = read_fp2platform(path)
    out = dict(zip(df["fingerprint"], df["platform"]))
    for fp, p in list(out.items()):
        out.setdefault(normalize(fp), p)
    for p in set(df["platform"]):                 # a spelling equal to a platform name maps to itself
        out.setdefault(p, p)
        out.setdefault(normalize(p), p)
    return out


# ------------------------------------------------------------------------------------------------- specs skeleton
def _blank_specs_row(platform: str) -> dict:
    row = {c: "" for c in PLATFORM_SPECS_COLUMNS}
    row["platform"] = platform
    row["spec_confidence"] = "C"
    return row


def skeleton_specs(mapping: pd.DataFrame, platforms: pd.DataFrame, platform_models: dict[str, tuple[str, str, str, str]],
                   ) -> pd.DataFrame:
    """One row per platform referenced by `mapping` (sorted). brand/model/body/size_code come from
    `platform_models` (platform -> (brand, model, body, size_code)); mass/wheelbase from CarSpecs when available;
    generation_note = CarDocs strings; tracks empty; spec_confidence 'C'."""
    pl = platforms.set_index("platform") if len(platforms) else pd.DataFrame()
    rows = []
    for plat in sorted(set(mapping["platform"])):
        row = _blank_specs_row(plat)
        if plat in platform_models:
            row["brand"], row["model"], row["body"], _size = platform_models[plat]
        if len(pl) and plat in pl.index:
            r = pl.loc[plat]
            if pd.notna(r.get("mass_kg")):
                row["curb_mass_kg"] = f"{float(r['mass_kg']):.0f}"
                row["mass_source"] = OPENDBC_SOURCE
            if pd.notna(r.get("wheelbase_mm")):
                row["wheelbase_mm"] = f"{float(r['wheelbase_mm']):.0f}"
                row["wheelbase_source"] = OPENDBC_SOURCE
            row["generation_note"] = str(r.get("docs") or "")
        rows.append(row)
    return pd.DataFrame(rows, columns=list(PLATFORM_SPECS_COLUMNS))


def merge_specs(existing: pd.DataFrame | None, skeleton: pd.DataFrame) -> pd.DataFrame:
    """Union of rows by platform; filled cells of `existing` always win, empty cells take the skeleton value.
    Rows only present in `existing` are kept (never deleted). spec_confidence defaults to 'C'."""
    if existing is None or not len(existing):
        out = skeleton.copy()
    else:
        ex = existing.copy().astype(str).replace({"nan": "", "None": ""})
        for c in PLATFORM_SPECS_COLUMNS:
            if c not in ex.columns:
                ex[c] = ""
        ex = ex[list(PLATFORM_SPECS_COLUMNS)].set_index("platform")
        sk = skeleton.set_index("platform")
        idx = sorted(set(ex.index) | set(sk.index))
        out = pd.DataFrame(index=idx, columns=[c for c in PLATFORM_SPECS_COLUMNS if c != "platform"]).fillna("")
        for c in out.columns:
            e = ex[c].reindex(idx).fillna("").astype(str)
            s = sk[c].reindex(idx).fillna("").astype(str) if c in sk.columns else pd.Series("", index=idx)
            out[c] = np.where(e.str.strip() != "", e, s)
        out = out.reset_index().rename(columns={"index": "platform"})[list(PLATFORM_SPECS_COLUMNS)]
    out["spec_confidence"] = out["spec_confidence"].replace("", "C")
    return out.sort_values("platform").reset_index(drop=True)


def read_platform_specs_raw(path: Path = PLATFORM_SPECS_CSV) -> pd.DataFrame:
    """platform_specs.csv as written (every cell a string, '' when empty); empty frame when the file is absent."""
    if not Path(path).exists():
        return pd.DataFrame(columns=list(PLATFORM_SPECS_COLUMNS))
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in PLATFORM_SPECS_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[list(PLATFORM_SPECS_COLUMNS)].copy()


def read_platform_specs(path: Path = PLATFORM_SPECS_CSV) -> pd.DataFrame:
    """platform_specs.csv with numeric spec columns as float (NaN when empty)."""
    df = read_platform_specs_raw(path)
    for c in SPEC_NUMERIC:
        df[c] = pd.to_numeric(df[c].where(df[c].str.strip() != "", None), errors="coerce")
    return df


def write_csv_sorted(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path
