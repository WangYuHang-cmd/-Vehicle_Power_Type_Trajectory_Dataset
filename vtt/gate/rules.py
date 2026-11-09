"""Gate rules: leakage patterns for text and column names, table specs of the release, enum domains and masking.

Every text rule returns a COUNT of hits and never the matched text; whatever a finding carries (path, column,
detail) goes through `gate_mask` first. Hard rules cannot be allow-listed. The allow-list (gate_allow.json next to
this file) covers only two things: ISO dates per file and the single public contact e-mail address.

Pre-blanking before the hex rules: floats with a decimal point are blanked in every text (ids never contain '.'),
unquoted JSON integers of >= 12 digits in numeric context are blanked in JSON text only (long counts, never ids);
a bare digits-only 16-char token in prose or CSV is still reported.

Several pattern literals below are assembled from pieces on purpose: this file is itself scanned by the pre-push
gate, and a literal coordinate word, path prefix or driver-assistance acronym in the source would be a hit.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vtt import config as C
from vtt import private as PV
from vtt import schema as S

ALLOW_PATH = Path(__file__).with_name("gate_allow.json")

# ------------------------------------------------------------------------------------------------ text patterns
HEX16 = PV.HEX16                       # 16 hex chars (incl. all-digit) between non-alphanumerics
HEX12 = PV.HEX12                       # 12 hex chars: the internal salted uids (route_uid, identity_uid)
ROUTE_DATE = PV.ROUTE_DATE             # YYYY-MM-DD--HH-MM-SS
ROUTE_NUM = PV.ROUTE_NUM               # <8 hex>--<10 hex>
DECIMAL = re.compile(r"(?<![0-9a-zA-Z])-?\d+\.\d+(?:[eE][-+]?\d+)?(?![0-9a-zA-Z])")
JSON_LONG_INT = re.compile(r"(?<=[:\[,\s])-?\d{12,}(?=\s*[,}\]])")
ISO_DATE = re.compile(r"(?<![0-9])\d{4}-\d{2}-\d{2}(?![0-9])")
URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")      # spec-sheet / press-kit links carry hex-like ids and dates of their own


def blank_urls(text: str) -> str:
    """Remove http(s) URLs before the shape-based rules run (the exact private-token match still sees the raw text)."""
    return URL_RE.sub(" ", text)

PATH_PREFIXES = tuple("/" + w + "/" for w in ("home", "data", "mnt", "tmp"))
# absolute prefixes only: a prefix preceded by a word character, "$", "}", ">" or "." is a relative fragment or a
# placeholder path (a build sub-folder, "<root>/...", "$D/...", "./..."), never a local absolute path
PATHS = re.compile(r"(?<![A-Za-z0-9_$}>.])(?:" + "|".join(re.escape(p) for p in PATH_PREFIXES) + ")")
# coordinate / origin words (assembled: see module docstring)
COORD_WORDS = tuple(w.replace("-", "") for w in ("la-t", "lo-n", "ln-g", "lati-tude", "longi-tude", "gp-s", "ge-o",
                                                 "coun-try", "us_-state", "in_-house"))
_COORD_SHORT = frozenset(COORD_WORDS[:2])                     # the two 3-letter words that also mean 'lateral' / 'longitudinal'
_COORD_LONG = frozenset(COORD_WORDS[2:8])
_COORD_COMPOUND = tuple(COORD_WORDS[8:])
COORD_TEXT = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(COORD_WORDS) + r")(?![A-Za-z0-9_])", re.I)
LATERAL_NEXT = frozenset({"acc", "accel", "active", "frac", "jerk"})     # *_acc_mps2, op_*_active, op_*_frac are lateral dynamics
VIN_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[A-HJ-NPR-Za-hj-npr-z0-9]{17}(?![A-Za-z0-9_])")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ANON_AUTH = re.compile(r"anonymous\s+authors?", re.I)
_ACRONYM = "AD" + "AS"                                        # driver-assistance acronym (never written literally here)
ACRONYM_WORD = re.compile(r"(?<![A-Za-z0-9_])" + _ACRONYM + r"(?![A-Za-z0-9_])", re.I)
CITATION_TOKEN = re.compile(_ACRONYM + "-TO")                 # the only allowed use: the citation token
BIBTEX_FENCE = re.compile(r"```bibtex[ \t]*\n.*?\n```", re.S)
MONO_COL = re.compile(r"mono|_ns$|timestamp", re.I)
VIN_COL = re.compile(r"(^|_)vin($|_)", re.I)
DENY_COL = re.compile(r"_uid$|^uid$|git|commit|version|fingerprint|dongle|route_id|rel_path|salt|identity", re.I)
JSON_KEY = re.compile(r'"([A-Za-z0-9_\-. ]+)"\s*:')
TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./@:-]*[A-Za-z0-9_]|[A-Za-z0-9_]")
ID_FORMATS = {"vehicle_id": re.compile(r"vehicle_\d{4}"), "drive_id": re.compile(r"drive_\d{6}"),
              "group_id": re.compile(r"group_\d{4}"), "window_id": re.compile(r"drive_\d{6}_w\d+")}

HARD_RULES = ("HEX16", "HEX12", "ROUTE_DATE", "ROUTE_NUM", "PATHS", "COORD", "VIN", "MONO", "ANON_AUTH", "ACRONYM",
              "FORBIDDEN", "EXACT")
SOFT_RULES = ("ISO_DATE", "EMAIL")       # allow-listable (per file / the one address)

# ------------------------------------------------------------------------------------------------ table rules
# derived release tables that have no spec in vtt.schema (written by vtt.package, checked by the gate)
VEHICLES_COLUMNS: list[tuple[str, str, str]] = [
    ("vehicle_id", "str", "pseudonymous vehicle (device x platform)"),
    ("group_id", "str", "identity cluster (unit of all splits)"),
    ("platform", "str", "canonical vehicle platform"), ("brand", "str", ""), ("model", "str", ""), ("body", "str", ""),
    ("size_class", "str", ""), ("powertrain", "str", ""), ("powertrain_raw", "str", ""), ("region", "str", ""),
    ("n_drives", "int32", "drives of this vehicle"), ("duration_h", "float32", "nominal hours"),
    ("moving_h", "float32", "hours with v >= 1 m/s"), ("das_any_h", "float32", "hours with any automation activity"),
    ("n_windows_eligible", "int32", "eligible windows"),
    ("split_device", "str", ""), ("split_platform", "str", ""), ("low_support_cell", "bool", ""),
]
PLATFORMS_COLUMNS: list[tuple[str, str, str]] = [
    ("platform", "str", "canonical vehicle platform"), ("brand", "str", ""), ("model", "str", ""), ("body", "str", ""),
    ("size_code", "str", ""), ("wheelbase_mm", "float32", ""), ("track_front_mm", "float32", ""), ("track_rear_mm", "float32", ""),
    ("footprint_m2", "float32", ""), ("curb_mass_kg", "float32", ""), ("size_class", "str", ""),
    ("powertrain", "str", ""), ("powertrain_raw", "str", ""),
    ("n_vehicles", "int32", ""), ("n_groups", "int32", ""), ("n_drives", "int32", ""), ("duration_h", "float32", ""),
    ("n_windows_eligible", "int32", ""),
    ("split_platform", "str", "split of the platform's drives: train / validation / test; 'mixed' when its groups fall in several "
                              "splits (validation is group-disjoint inside the non-held-out pool, so train+validation is expected; "
                              "'test' is exclusive), 'excluded' when every drive of the platform is excluded"),
]
TABLE_SPECS = {"drive": S.DRIVE_COLUMNS, "drives": S.DRIVES_INDEX_COLUMNS, "windows": S.WINDOWS_INDEX_COLUMNS,
               "vehicles": VEHICLES_COLUMNS, "platforms": PLATFORMS_COLUMNS}
SHARD_TABLE = "windows_shard"
SHARD_INDEX_COLUMNS = S.WINDOWS_INDEX_COLUMNS
SHARD_SEQ_COLUMNS = tuple(C.OFFICIAL_INPUTS)          # one fixed_size_list<float32>[WINDOW_SAMPLES] per official input
# flexible tables (splits/*.parquet, index/split_summary.csv): names must be in this set and not match DENY_COL
SPLIT_TABLE_ALLOW = frozenset(S.names(S.WINDOWS_INDEX_COLUMNS)) | frozenset(S.names(S.DRIVES_INDEX_COLUMNS)) | frozenset({
    "weight", "task", "split", "class", "label", "cell", "protocol", "rank", "order", "hours", "windows", "groups",
    "platforms", "drives", "vehicles", "n_windows", "n_groups", "n_platforms", "n_drives", "n_vehicles", "share",
    "eval_hours", "train_hours", "count", "supported", "low_support", "max_group_share", "max_platform_share",
    "n_classes", "min_class_windows", "feasible", "note"})
COUNTER_ALLOW = frozenset({"n_rows", "row0", "row1", "w_idx", "n_drives", "n_vehicles", "n_groups", "n_windows",
                           "n_platforms", "n_windows_eligible", "windows", "groups", "platforms", "drives", "vehicles",
                           "count", "rank", "order", "n_segments", "n_seg_rlog_dec", "n_gaps", "gap_count",
                           "n_accel_ep", "n_decel_ep", "n_lead_accel_events"})
MAX_ABS_NUMERIC = 1e9                   # a boot clock (ns) or unix time (s) is far above; physical values far below
SPLIT_VALUES = frozenset({"train", "validation", "test"})
ENUM_DOMAINS: dict[str, frozenset] = {
    "size_class": frozenset(C.SIZE_CLASSES),
    "powertrain": frozenset(C.POWERTRAIN4),
    "powertrain_raw": frozenset(C.POWERTRAIN_MAP),
    "software_era": frozenset(C.SOFTWARE_ERAS),
    "region": frozenset(C.REGION_BUCKETS.values()) | {C.REGION_OTHER},
    "pose_source": frozenset({"livePose", "liveLocationKalman", "mixed", "none", 0, 1, 2}),
    "lead_source": frozenset({0, 1, 2}),
    "split_device": SPLIT_VALUES, "split_platform": SPLIT_VALUES | {"mixed", "excluded"},   # excluded: S8 contract note
    "kind": frozenset({"human", "das", "mixed"}),
}
MAX_VEHICLES = 2000
MAX_DRIVES = 12000


# ------------------------------------------------------------------------------------------------ findings
@dataclass
class Finding:
    """One gate finding. `path`, `column` and `detail` are already masked; `count` = number of hits."""
    gate: str
    rule: str
    path: str = ""
    column: str = ""
    count: int = 1
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    utc: str
    root: str
    n_files: int = 0
    n_bytes: int = 0
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            key = f"{f.gate}/{f.rule}"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def to_dict(self) -> dict:
        return {"utc": self.utc, "root": gate_mask(self.root), "n_files": self.n_files, "n_bytes": self.n_bytes,
                "n_findings": len(self.findings), "counts": self.counts(), "ok": self.ok,
                "findings": [f.to_dict() for f in self.findings], "warnings": [gate_mask(w) for w in self.warnings]}


def gate_mask(text) -> str:
    """vtt.private.mask plus the gate's own placeholders (12-hex uid, e-mail, VIN-shaped token)."""
    out = PV.mask(str(text))
    out = HEX12.sub("<hex12>", out)
    out = EMAIL.sub("<email>", out)
    out = VIN_TOKEN.sub(lambda m: "<vin>" if _vin_like(m.group(0)) else m.group(0), out)
    return out


def load_allow(path: Path | None = None) -> dict:
    """gate_allow.json: {'email': str, 'iso_date_files': [...], 'pre_push_iso_date_files': [...],
    'history_accepted': [{'blob': <40-hex git blob sha>, 'rule': str, 'reason': str}, ...]} (see
    pre_push.accepted_history: reviewed findings in already-published history, bound to the immutable blob)."""
    p = Path(path) if path else ALLOW_PATH
    allow = json.loads(p.read_text()) if p.exists() else {}
    allow.setdefault("email", "")
    allow.setdefault("iso_date_files", [])
    allow.setdefault("pre_push_iso_date_files", [])
    allow.setdefault("history_accepted", [])
    return allow


# ------------------------------------------------------------------------------------------------ text rules
def blank_numbers(text: str, is_json: bool = False) -> str:
    """Blank decimal floats everywhere and >= 12-digit unquoted JSON integers in JSON text (see module docstring)."""
    out = DECIMAL.sub(" ", text)
    if is_json:
        out = JSON_LONG_INT.sub(" 0", out)
    return out


def hits_hex16(text: str, is_json: bool = False) -> int:
    return len(HEX16.findall(blank_numbers(text, is_json)))


def hits_hex12(text: str, is_json: bool = False) -> int:
    return len(HEX12.findall(blank_numbers(text, is_json)))


def hits_route_date(text: str) -> int:
    return len(ROUTE_DATE.findall(text))


def hits_route_num(text: str) -> int:
    return len(ROUTE_NUM.findall(text))


def hits_iso_date(text: str) -> int:
    """ISO dates that are not part of a route date (those count under ROUTE_DATE)."""
    return len(ISO_DATE.findall(ROUTE_DATE.sub(" ", text)))


def hits_paths(text: str) -> int:
    """Absolute local path prefixes (PATH_PREFIXES) at the start of a path; relative fragments and placeholder
    paths (see PATHS) do not count."""
    return len(PATHS.findall(text))


def hits_coord(text: str) -> int:
    """Coordinate / origin words (COORD_WORDS) as whole words, any case."""
    return len(COORD_TEXT.findall(text))


def _vin_like(tok: str) -> bool:
    return any(ch.isalpha() for ch in tok) and any(ch.isdigit() for ch in tok)


def hits_vin(text: str) -> int:
    """17-char tokens over the VIN alphabet (no I, O, Q) with at least one letter and one digit."""
    return sum(1 for tok in VIN_TOKEN.findall(text) if _vin_like(tok))


def hits_email(text: str, allowed: str = "") -> int:
    """E-mail addresses other than the single allowed contact address."""
    return sum(1 for m in EMAIL.findall(text) if not allowed or m.lower() != allowed.lower())


def hits_anon_auth(text: str) -> int:
    """The double-blind author placeholder phrase, any case."""
    return len(ANON_AUTH.findall(text))


def hits_acronym(text: str, readme: bool = False) -> int:
    """The driver-assistance acronym outside the citation token; in a README the single bibtex fence is exempt
    (a second bibtex fence counts as a hit)."""
    t = text
    if readme:
        fences = BIBTEX_FENCE.findall(t)
        if len(fences) > 1:
            return len(fences)
        t = BIBTEX_FENCE.sub(" ", t)
    t = CITATION_TOKEN.sub(" ", t)
    return len(ACRONYM_WORD.findall(t))


def hits_forbidden(text: str, tokens) -> int:
    """Case-insensitive hits of the private forbidden tokens (word-bounded for plain tokens, substring otherwise)."""
    if not tokens:
        return 0
    low = text.lower()
    n = 0
    for tok in tokens:
        t = str(tok).strip().lower()
        if not t:
            continue
        if re.fullmatch(r"[a-z0-9_]+", t):
            n += len(re.findall(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low))
        else:
            n += low.count(t)
    return n


def hits_exact(text: str, private: frozenset | set | None) -> int:
    """Exact (case-insensitive) matches of tokens of the private token set, incl. the '/'-separated parts of paths
    (the set itself never holds a car-model folder name, see `private_tokens`)."""
    if not private:
        return 0
    n = 0
    for tok in TOKEN.findall(text):
        low = tok.lower()
        if low in private:
            n += 1
        elif "/" in low:
            n += sum(1 for part in low.split("/") if part in private)
    return n


def scan_text(text: str, *, is_json: bool = False, readme: bool = False, allowed_email: str = "",
              iso_allowed: bool = False, forbidden=None, private=None) -> dict[str, int]:
    """All text rules on one string; returns {rule: hits} with zero-hit rules dropped."""
    t = blank_urls(text)                      # shape rules ignore URLs; exact/forbidden matching below uses the raw text
    hits = {
        "HEX16": hits_hex16(t, is_json), "HEX12": hits_hex12(t, is_json),
        "ROUTE_DATE": hits_route_date(t), "ROUTE_NUM": hits_route_num(t),
        "ISO_DATE": 0 if iso_allowed else hits_iso_date(t),
        "PATHS": hits_paths(t), "COORD": hits_coord(t), "VIN": hits_vin(t),
        "EMAIL": hits_email(text, allowed_email), "ANON_AUTH": hits_anon_auth(text), "ACRONYM": hits_acronym(text, readme),
        "FORBIDDEN": hits_forbidden(text, forbidden), "EXACT": hits_exact(text, private),
    }
    return {k: v for k, v in hits.items() if v}


# ------------------------------------------------------------------------------------------------ name rules
def col_coord(name: str) -> bool:
    """Coordinate-like column name; the 3-letter words followed by a lateral-dynamics token (LATERAL_NEXT) are exempt."""
    low = str(name).lower()
    if any(w in low for w in _COORD_COMPOUND):
        return True
    toks = low.split("_")
    for i, tok in enumerate(toks):
        if tok in _COORD_LONG:
            return True
        if tok in _COORD_SHORT and not (i + 1 < len(toks) and toks[i + 1] in LATERAL_NEXT):
            return True
    return False


def col_vin(name: str) -> bool:
    return VIN_COL.search(str(name)) is not None


def col_mono(name: str) -> bool:
    """Boot-clock / timestamp column names (MONO_COL)."""
    return MONO_COL.search(str(name)) is not None


def col_denied(name: str) -> bool:
    """Names forbidden in the flexible tables (DENY_COL: uids, git, version, fingerprint, dongle, route_id, ...)."""
    return DENY_COL.search(str(name)) is not None


def name_rule_hits(names) -> dict[str, int]:
    """Column-name rules only (MONO, COORD, VIN) over a list of names / JSON keys."""
    hits = {"MONO": sum(1 for n in names if col_mono(n)), "COORD": sum(1 for n in names if col_coord(n)),
            "VIN": sum(1 for n in names if col_vin(n))}
    return {k: v for k, v in hits.items() if v}


def scan_names(names, *, forbidden=None, private=None) -> dict[str, int]:
    """Column names: the text rules on the joined names plus the column-name rules."""
    hits = scan_text("\n".join(str(n) for n in names), forbidden=forbidden, private=private)
    for k, v in name_rule_hits(names).items():        # a name caught by both views counts once
        hits[k] = max(hits.get(k, 0), v)
    return hits


def json_keys(text: str) -> list[str]:
    return JSON_KEY.findall(text)


def id_format_ok(column: str, value) -> bool:
    """Pseudonym format of vehicle_id / drive_id / group_id / window_id values (other columns always pass)."""
    rx = ID_FORMATS.get(column)
    if rx is None or value is None:
        return True
    return rx.fullmatch(str(value)) is not None


# ------------------------------------------------------------------------------------------------ private tokens
def private_readable(private_root: Path | None) -> bool:
    try:
        return private_root is not None and Path(private_root).is_dir() and bool(list(Path(private_root).iterdir()))
    except OSError:
        return False


PATH_COLS = frozenset({"rel_path", "copies_rel"})      # car_dir/dongle_dir/route_id strings (copies_rel: a JSON list)
ID_COLS = frozenset({"route_uid", "dongle_dir", "dongle_log", "route_id", "identity_uid", "vin_hash"}) | PATH_COLS


def _path_items(s: str) -> list[str]:
    """A rel_path string or a JSON list of them (copies_rel) as a list of path strings."""
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s) if x]
        except ValueError:
            return [s]
    return [s]


_ID_LIKE_RES = tuple(re.compile(x) for x in (r"[0-9a-f]{16}", r"[0-9a-f]{12}", r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}",
                                            r"[0-9a-f]{8}--[0-9a-f]{10}"))


def _id_like(s: str) -> bool:
    """A route folder or deeper path component that can be an identifier: carries a digit or has the exact shape of a
    device / route / uid token (letter-only hex shapes included, as synthetic fixtures use them)."""
    return any(ch.isdigit() for ch in s) or any(r.fullmatch(s.lower()) for r in _ID_LIKE_RES)


STRUCTURAL_MIN_PATHS = 20      # an alphabetic path component found in this many paths is layout, not an identifier


def private_tokens(private_root: Path | None, min_len: int = 4) -> frozenset | None:
    """Lower-cased exact tokens that must never appear in a public tree, from every readable private file:
    salt, dongle_log/dongle_dir/route_id/route_uid/identity_uid/vin_hash (uid_map, routes_resolved, pseudonym_map,
    labels_private), rel_path / copies_rel (the whole path plus its dongle-folder and route components, never the
    first component: the car-model folder is a public platform spelling), misfiled folder names, lab dongles,
    forbidden tokens, identity merges. None when the private root is not readable (the caller WARNs). Never logs
    or prints any token."""
    if not private_readable(private_root):
        return None
    from collections import Counter
    import pandas as pd
    root = Path(private_root)
    toks: set[str] = set()
    nonpath: set[str] = set()            # tokens that came from an identifier column (never dropped)
    path_alpha: Counter = Counter()      # purely alphabetic path components -> number of path strings carrying them

    def keep(s: str) -> bool:
        return bool(s) and s.lower() not in ("nan", "none", "null") and len(s) >= min_len

    def add(values, path_like: bool = False, column: str = "") -> None:
        for v in values:
            if v is None:
                continue
            s = str(v).strip()
            if not keep(s):
                continue
            if not path_like:
                if column == "route_id" and not _id_like(s):
                    continue                        # a route folder with no identifier shape is layout (a stray log sub-folder), not a route id
                toks.add(s.lower())
                nonpath.add(s.lower())
                continue
            for item in _path_items(s):
                item = item.strip().strip("/")
                if keep(item):
                    toks.add(item.lower())
                parts = [q for q in item.split("/") if q]
                tail = parts[1:] if len(parts) >= 3 else parts        # drop the car-model folder, keep dongle dir + route id
                alpha_here: set[str] = set()
                for j, q in enumerate(tail):
                    if not keep(q):
                        continue
                    if j > 0 and not _id_like(q):
                        continue                    # route ids and deeper components have identifier shapes; plain words are layout
                    toks.add(q.lower())
                    if q.isalpha():
                        alpha_here.add(q.lower())   # a dongle-position folder (person name or hex) - counted for the frequency rule
                path_alpha.update(alpha_here)

    for name in ("salt.txt", "seeds/salt.txt", "forbidden_tokens.txt", "release_seed.txt"):
        p = root / name
        if p.exists():
            add(line for line in p.read_text().splitlines() if line.strip() and not line.startswith("#"))
    for name in ("uid_map.parquet", "routes_resolved.parquet", "pseudonym_map.parquet", "labels_private.parquet"):
        p = root / name
        if p.exists():
            try:
                df = pd.read_parquet(p)
            except Exception:                      # noqa: BLE001 - an unreadable private table is not a public finding
                continue
            for c in df.columns:
                if c in ID_COLS:
                    add(df[c].dropna().astype(str).tolist(), path_like=c in PATH_COLS, column=c)
    for name in ("inhouse_groups.csv", "identity_merges.csv", "exclude_routes.csv", "raw_manifest.csv"):
        p = root / name
        if p.exists():
            try:
                df = pd.read_csv(p, dtype=str, keep_default_na=False)
            except Exception:                      # noqa: BLE001
                continue
            for c in df.columns:
                if c in ID_COLS:
                    add(df[c].tolist(), path_like=c in PATH_COLS, column=c)
    p = root / "misfiled_dirs.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            add(list(d.keys()) + [v for v in d.values() if isinstance(v, str)])
        except Exception:                          # noqa: BLE001
            pass
    try:
        for k, v in PV.misfiled_dirs(root).items():
            add([k, v])
    except Exception:                              # noqa: BLE001
        pass
    # structural folder / file names of the raw library (purely alphabetic components shared by many paths, e.g. the
    # log sub-folder) are layout, not identifiers; identifier columns and rare components are never dropped
    for tok, n in path_alpha.items():
        if n >= STRUCTURAL_MIN_PATHS and tok not in nonpath:
            toks.discard(tok)
    toks.discard("")
    return frozenset(toks)
