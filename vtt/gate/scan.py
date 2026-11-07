"""Release-tree scanner (pre-upload gate).

G0 path allow-list and required files, G1 names + text files, G2 parquet / CSV string values, G3 exact column sets,
G4 parquet metadata keys, G5 numeric ranges (no boot clocks; t_s from 0, strictly increasing), G6 enum domains,
G8 cardinality and pseudonym formats, G9 exact private tokens (when the private root is readable), and the
CHECKSUMS.sha256 regeneration equality. Findings are aggregated per (gate, rule, file[, column]) with a hit count,
all paths / columns / details masked. Exit status is decided by the caller: 0 iff `Report.ok`.

Parquet files are read in row batches of about SCAN_BATCH_BYTES (never as a whole table): numeric, sequence and t_s
checks see every row of every file; the distinct string values of the index and splits tables are checked in full,
those of the drive tables and window shards on their first `sample_rows` rows (SAMPLED_TABLES).
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vtt import config as C
from vtt import private as PV
from vtt import schema as S
from vtt.gate import rules as R
from vtt.gate.rules import Finding, Report, gate_mask

ALLOWED_PATTERNS = tuple(re.compile(p) for p in (
    r"README\.md", r"LICENSE", r"CHECKSUMS\.sha256", r"\.gitattributes", r"\.gate_ok",
    r"index/[a-z0-9_]+\.(?:parquet|csv|json)", r"splits/[a-z0-9_]+\.(?:parquet|json)", r"schema/[a-z0-9_]+\.json",
    r"data/vehicle_\d{4}/drive_\d{6}\.parquet", r"windows/(?:train|validation|test)/[a-z0-9_\-]+\.parquet"))
SKIP_DIRS = (".cache",)                                   # HF upload cache: never scanned, never check-summed
CHECKSUM_EXCLUDE = frozenset({"CHECKSUMS.sha256", ".gate_ok"})
REQUIRED_FILES = ("README.md", "LICENSE", "CHECKSUMS.sha256", "index/drives.parquet")
TEXT_SUFFIXES = frozenset({".md", ".json", ".csv", ".txt", ".sha256", ".yaml", ".yml", ""})
SPEC_FILES = {"index/drives.parquet": "drives", "index/windows_60s.parquet": "windows",
              "index/vehicles.parquet": "vehicles", "index/platforms.parquet": "platforms"}
DRIVE_FILE = re.compile(r"data/vehicle_\d{4}/drive_\d{6}\.parquet")
SCAN_BATCH_BYTES = 32 * 1024 * 1024                        # target size of one row batch of the parquet scan
SAMPLED_TABLES = frozenset({"drive", R.SHARD_TABLE})       # string values checked on the first `sample_rows` rows only


def utc_stamp() -> str:
    """Compact UTC stamp for file names (no ISO-date pattern)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def is_allowed(rel: str) -> bool:
    return any(p.fullmatch(rel) for p in ALLOWED_PATTERNS)


def list_release_files(root: Path) -> list[str]:
    """Sorted relative POSIX paths of every file under root, skipping SKIP_DIRS."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in filenames:
            out.append(PurePosixPath(Path(dirpath, f).relative_to(root)).as_posix())
    return sorted(out)


def classify(rel: str) -> tuple[str, str]:
    """(kind, table): kind in spec | shard | flex | csv | text | other."""
    pp = PurePosixPath(rel)
    if rel.startswith("data/") and rel.endswith(".parquet"):
        return "spec", "drive"
    if rel in SPEC_FILES:
        return "spec", SPEC_FILES[rel]
    if rel.startswith("windows/") and rel.endswith(".parquet"):
        return "shard", R.SHARD_TABLE
    if rel.endswith(".parquet"):
        return "flex", pp.stem
    if rel.endswith(".csv"):
        return "csv", pp.stem
    if pp.suffix in TEXT_SUFFIXES or pp.name in ("LICENSE", ".gitattributes", ".gate_ok"):
        return "text", ""
    return "other", ""


class Ctx:
    """Scan context handed to the workers (picklable: paths, allow dict, token lists / sets)."""

    def __init__(self, root: Path, allow: dict, forbidden: list[str], private, sample_rows: int) -> None:
        self.root = Path(root)
        self.allow = allow
        self.forbidden = forbidden
        self.private = private
        self.sample_rows = sample_rows


_CTX: Ctx | None = None


def _init(ctx: Ctx) -> None:
    global _CTX
    _CTX = ctx


# ------------------------------------------------------------------------------------------------ per-file checks
def check_text(p: Path, rel: str, ctx: Ctx) -> tuple[list[Finding], list[str]]:
    text = p.read_text(errors="replace")
    is_json = rel.endswith(".json")
    readme = PurePosixPath(rel).name.startswith("README")
    hits = R.scan_text(text, is_json=is_json, readme=readme, allowed_email=ctx.allow["email"],
                       iso_allowed=rel in ctx.allow["iso_date_files"], forbidden=ctx.forbidden, private=ctx.private)
    findings = [Finding("G1", rule, gate_mask(rel), count=n) for rule, n in hits.items()]
    if is_json:
        for rule, n in R.name_rule_hits(R.json_keys(text)).items():
            findings.append(Finding("G1", rule, gate_mask(rel), column="<json_keys>", count=n))
    return findings, []


def _columns_finding(rel: str, names: list[str], expected: list[str]) -> list[Finding]:
    if names == expected:
        return []
    extra = [n for n in names if n not in expected]
    missing = [n for n in expected if n not in names]
    order = "ok" if sorted(names) == sorted(expected) else "differs"
    detail = f"{len(extra)} extra, {len(missing)} missing, order {order}"
    if extra:
        detail += "; extra: " + gate_mask(", ".join(extra[:5]))
    if missing:
        detail += "; missing: " + ", ".join(missing[:5])
    return [Finding("G3", "COLUMNS", gate_mask(rel), detail=detail)]


def _dtype_warnings(rel: str, sch: pa.Schema, spec) -> list[str]:
    out = []
    for name, dt in S.dtypes(spec).items():
        if name in sch.names and sch.field(name).type != S._PA[dt]:
            out.append(f"{rel}: column {name} is {sch.field(name).type}, spec {dt}")
    return out


def _value_type(t: pa.DataType) -> pa.DataType:
    return t.value_type if pa.types.is_dictionary(t) else t


def _row_bytes(sch: pa.Schema) -> int:
    n = 0
    for f in sch:
        t = _value_type(f.type)
        if pa.types.is_fixed_size_list(t):
            try:
                n += t.list_size * max(1, t.value_type.bit_width // 8)
            except ValueError:
                n += t.list_size * 8
        elif pa.types.is_list(t) or pa.types.is_large_list(t):
            n += 64
        elif pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t):
            n += 32
        else:
            try:
                n += max(1, t.bit_width // 8)
            except ValueError:
                n += 8
    return max(1, n)


def batch_rows(sch: pa.Schema, target_bytes: int | None = None) -> int:
    """Rows per batch so that one batch holds about `target_bytes` (default SCAN_BATCH_BYTES; 256 .. 65536)."""
    target = SCAN_BATCH_BYTES if target_bytes is None else target_bytes
    return int(min(65536, max(256, target // _row_bytes(sch))))


class _Acc:
    """Streaming accumulators of one parquet file over its row batches: distinct string values (all rows, or the
    first `sample_rows` when given), max |x| of numeric and sequence columns, values of numeric enum columns,
    sequence lengths and the t_s start / strict monotonicity."""

    def __init__(self, sch: pa.Schema, sample_rows: int | None) -> None:
        self.sch = sch
        self.sample_rows = sample_rows
        self.str_idx: list[int] = []
        self.num_idx: list[int] = []
        self.list_idx: list[int] = []
        for i, f in enumerate(sch):
            t = _value_type(f.type)
            if pa.types.is_string(t) or pa.types.is_large_string(t):
                self.str_idx.append(i)
            elif pa.types.is_integer(t) or pa.types.is_floating(t):
                self.num_idx.append(i)
            elif pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
                self.list_idx.append(i)
        self.uniq: dict[int, set] = {i: set() for i in self.str_idx}
        self.max_abs: dict[int, float] = {}
        self.enum_vals: dict[int, set] = {i: set() for i in self.num_idx if sch.field(i).name in R.ENUM_DOMAINS}
        self.lengths: dict[int, set] = {i: set() for i in self.list_idx}
        self.t_idx = next((i for i in self.num_idx if sch.field(i).name == "t_s"), -1)
        self.t_first: float | None = None
        self.t_prev: float | None = None
        self.t_mono = True
        self.rows = 0

    def _bump(self, i: int, arr) -> None:
        m = pc.max(pc.abs(arr)).as_py()                   # NaN ignored; all-NaN gives -inf, all-null None
        if m is not None:
            m = float(m)
            if m == m and (i not in self.max_abs or m > self.max_abs[i]):
                self.max_abs[i] = m

    def add(self, batch: pa.RecordBatch) -> None:
        n = batch.num_rows
        for i in self.str_idx:
            if self.sample_rows is not None and self.rows >= self.sample_rows:
                break
            col = batch.column(i)
            if pa.types.is_dictionary(col.type):
                col = col.dictionary_decode()
            if self.sample_rows is not None and self.rows + n > self.sample_rows:
                col = col.slice(0, self.sample_rows - self.rows)
            self.uniq[i].update(v for v in pc.unique(col).to_pylist() if v is not None)
        for i in self.num_idx:
            col = batch.column(i)
            if pa.types.is_dictionary(col.type):
                col = col.dictionary_decode()
            self._bump(i, col)
            if i in self.enum_vals:
                self.enum_vals[i].update(v for v in pc.unique(col).to_pylist() if v is not None)
            if i == self.t_idx and n:
                arr = np.asarray(col.to_numpy(zero_copy_only=False), dtype=float)
                if self.t_first is None:
                    self.t_first = float(arr[0])
                if self.t_prev is not None and not arr[0] > self.t_prev:
                    self.t_mono = False
                if n > 1 and not bool(np.all(np.diff(arr) > 0)):
                    self.t_mono = False
                self.t_prev = float(arr[-1])
        for i in self.list_idx:
            col = batch.column(i)
            flat = pc.list_flatten(col)
            if pa.types.is_floating(flat.type) or pa.types.is_integer(flat.type):
                self._bump(i, flat)
            self.lengths[i].update(v for v in pc.unique(pc.list_value_length(col)).to_pylist() if v is not None)
        self.rows += n


def _check_strings(rel: str, name: str, uniq: list, ctx: Ctx) -> list[Finding]:
    """G2 text rules, G8 pseudonym format and G6 enum domain over the distinct values of one string column."""
    hits = R.scan_text("\n".join(map(str, uniq)), allowed_email=ctx.allow["email"], forbidden=ctx.forbidden,
                       private=ctx.private)
    findings = [Finding("G2", rule, gate_mask(rel), column=gate_mask(name), count=n) for rule, n in hits.items()]
    bad_fmt = sum(1 for v in uniq if not R.id_format_ok(name, v))
    if bad_fmt:
        findings.append(Finding("G8", "ID_FORMAT", gate_mask(rel), column=gate_mask(name), count=bad_fmt))
    if name in R.ENUM_DOMAINS:
        bad = [v for v in uniq if v not in R.ENUM_DOMAINS[name]]
        if bad:
            findings.append(Finding("G6", "ENUM", gate_mask(rel), column=name, count=len(bad),
                                    detail=gate_mask(", ".join(map(str, bad[:3])))))
    return findings


def _value_findings(rel: str, sch: pa.Schema, acc: _Acc, ctx: Ctx) -> list[Finding]:
    """G2 / G6 / G8 on strings, G5 magnitude + t_s, G6 numeric enums, G3 sequence lengths from the accumulators."""
    findings: list[Finding] = []
    for i in acc.str_idx:
        findings += _check_strings(rel, sch.field(i).name, sorted(acc.uniq[i]), ctx)
    for i in acc.num_idx:
        name = sch.field(i).name
        mx = acc.max_abs.get(i)
        if name not in R.COUNTER_ALLOW and mx is not None and mx >= R.MAX_ABS_NUMERIC:
            findings.append(Finding("G5", "MAGNITUDE", gate_mask(rel), column=gate_mask(name), detail=f"max_abs={mx:.4g}"))
        if i in acc.enum_vals:
            bad = sorted((v for v in acc.enum_vals[i] if v not in R.ENUM_DOMAINS[name]), key=str)
            if bad:
                findings.append(Finding("G6", "ENUM", gate_mask(rel), column=name, count=len(bad), detail=", ".join(map(str, bad[:3]))))
    if acc.t_idx >= 0 and acc.rows:
        start_ok = acc.t_first is not None and math.isfinite(acc.t_first) and acc.t_first == 0.0
        if not (start_ok and acc.t_mono):
            findings.append(Finding("G5", "T_S", gate_mask(rel), column="t_s",
                                    detail=f"start={acc.t_first:.4g}, strictly_increasing={acc.t_mono}"))
    for i in acc.list_idx:
        name = sch.field(i).name
        mx = acc.max_abs.get(i)
        if mx is not None and mx >= R.MAX_ABS_NUMERIC:
            findings.append(Finding("G5", "MAGNITUDE", gate_mask(rel), column=gate_mask(name), detail=f"max_abs={mx:.4g}"))
        lengths = sorted(acc.lengths[i])
        if lengths and lengths != [C.WINDOW_SAMPLES]:
            findings.append(Finding("G3", "SEQ_LEN", gate_mask(rel), column=gate_mask(name), detail=f"lengths={lengths[:4]}"))
    return findings


def check_parquet(p: Path, rel: str, kind: str, table: str, ctx: Ctx) -> tuple[list[Finding], list[str]]:
    """Schema checks (G4 metadata, G1 names, G3 columns) and one streaming pass over the rows (see _Acc)."""
    findings: list[Finding] = []
    warnings: list[str] = []
    pf = pq.ParquetFile(p)
    sch = pf.schema_arrow
    meta = dict(sch.metadata or {})
    if set(meta.keys()) != set(S.PARQUET_METADATA_KEYS):                                   # G4
        keys = ",".join(sorted(k.decode(errors="replace") for k in meta))
        findings.append(Finding("G4", "META_KEYS", gate_mask(rel), detail=gate_mask(keys)))
    else:
        got = meta[b"vtt_table"].decode(errors="replace")
        if kind in ("spec", "shard") and got != table:
            findings.append(Finding("G4", "META_TABLE", gate_mask(rel), detail=gate_mask(got)))
        if meta[b"vtt_schema_version"].decode(errors="replace") != S.SCHEMA_VERSION:
            warnings.append(f"{rel}: schema version differs from {S.SCHEMA_VERSION}")
    names = list(sch.names)
    for rule, n in R.scan_names(names, forbidden=ctx.forbidden, private=ctx.private).items():    # G1 on names
        findings.append(Finding("G1", rule, gate_mask(rel), column="<columns>", count=n))
    if kind == "spec":                                                                          # G3
        spec = R.TABLE_SPECS[table]
        findings += _columns_finding(rel, names, S.names(spec))
        warnings += _dtype_warnings(rel, sch, spec)
    elif kind == "shard":
        findings += _columns_finding(rel, names, S.names(R.SHARD_INDEX_COLUMNS) + list(R.SHARD_SEQ_COLUMNS))
        for c in R.SHARD_SEQ_COLUMNS:
            if c in names:
                t = sch.field(c).type
                ok = pa.types.is_fixed_size_list(t) and t.list_size == C.WINDOW_SAMPLES and pa.types.is_float32(t.value_type)
                if not ok:
                    findings.append(Finding("G3", "SEQ_TYPE", gate_mask(rel), column=c, detail=str(t)))
        warnings += _dtype_warnings(rel, sch, R.SHARD_INDEX_COLUMNS)
    else:
        bad = [n for n in names if n not in R.SPLIT_TABLE_ALLOW or R.col_denied(n)]
        if bad:
            findings.append(Finding("G3", "COLUMNS_ALLOW", gate_mask(rel), count=len(bad), detail=gate_mask(", ".join(bad[:5]))))
    acc = _Acc(sch, ctx.sample_rows if table in SAMPLED_TABLES else None)
    for batch in pf.iter_batches(batch_size=batch_rows(sch)):
        acc.add(batch)
    findings += _value_findings(rel, sch, acc, ctx)                                             # G2 / G5 / G6 / G8 / SEQ_LEN
    return findings, warnings


def check_csv(p: Path, rel: str, ctx: Ctx) -> tuple[list[Finding], list[str]]:
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    names = list(df.columns)
    findings = [Finding("G1", rule, gate_mask(rel), column="<columns>", count=n)
                for rule, n in R.scan_names(names, forbidden=ctx.forbidden, private=ctx.private).items()]
    bad = [n for n in names if n not in R.SPLIT_TABLE_ALLOW or R.col_denied(n)]
    if bad:
        findings.append(Finding("G3", "COLUMNS_ALLOW", gate_mask(rel), count=len(bad), detail=gate_mask(", ".join(bad[:5]))))
    for c in names:
        vals = sorted(set(df[c].tolist()) - {""})
        if not vals:
            continue
        num = pd.to_numeric(pd.Series(vals), errors="coerce")
        if num.notna().all():
            if c not in R.COUNTER_ALLOW and float(num.abs().max()) >= R.MAX_ABS_NUMERIC:
                findings.append(Finding("G5", "MAGNITUDE", gate_mask(rel), column=gate_mask(c), detail=f"max_abs={float(num.abs().max()):.4g}"))
            continue
        hits = R.scan_text("\n".join(vals), allowed_email=ctx.allow["email"], forbidden=ctx.forbidden, private=ctx.private)
        findings += [Finding("G2", rule, gate_mask(rel), column=gate_mask(c), count=n) for rule, n in hits.items()]
        if c in R.ENUM_DOMAINS:
            badv = [v for v in vals if v not in R.ENUM_DOMAINS[c]]
            if badv:
                findings.append(Finding("G6", "ENUM", gate_mask(rel), column=c, count=len(badv), detail=gate_mask(", ".join(badv[:3]))))
    return findings, []


def _check_one(rel: str):
    ctx = _CTX
    p = ctx.root / rel
    findings: list[Finding] = []
    warnings: list[str] = []
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    if not is_allowed(rel):
        findings.append(Finding("G0", "ALLOWLIST", gate_mask(rel)))
    for rule, n in R.scan_text(rel, allowed_email=ctx.allow["email"], forbidden=ctx.forbidden, private=ctx.private).items():
        findings.append(Finding("G1", rule, gate_mask(rel), column="<path>", count=n))
    kind, table = classify(rel)
    try:
        if kind in ("spec", "shard", "flex"):
            f, w = check_parquet(p, rel, kind, table, ctx)
        elif kind == "csv":
            f, w = check_csv(p, rel, ctx)
        elif kind == "text":
            f, w = check_text(p, rel, ctx)
        else:
            f, w = [], []
    except Exception as e:                        # noqa: BLE001 - an unreadable public file is itself a finding
        f, w = [Finding("G1", "UNREADABLE", gate_mask(rel), detail=type(e).__name__)], []
    return rel, findings + f, warnings + w, size


def _run(files: list[str], ctx: Ctx, workers: int):
    if workers <= 1 or len(files) < 4:
        _init(ctx)
        return [_check_one(f) for f in files]
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(ctx,)) as pool:   # spawn: no fork after threads
        return list(pool.imap(_check_one, files, chunksize=8))


# ------------------------------------------------------------------------------------------------ tree checks
def tree_checks(root: Path, files: list[str]) -> list[Finding]:
    findings = [Finding("G0", "MISSING", req) for req in REQUIRED_FILES if req not in files]
    idx = root / "index" / "drives.parquet"
    if idx.exists():
        try:
            sch = pq.read_schema(idx)
            cols = [c for c in ("vehicle_id", "drive_id", "file") if c in sch.names]
            t = pq.read_table(idx, columns=cols)
            if "vehicle_id" in cols and "drive_id" in cols:
                nv = len(pc.unique(t["vehicle_id"]))
                nd = len(pc.unique(t["drive_id"]))
                if nv > R.MAX_VEHICLES or nd > R.MAX_DRIVES:
                    findings.append(Finding("G8", "CARDINALITY", "index/drives.parquet", detail=f"vehicles={nv} drives={nd}"))
                if nd != t.num_rows:
                    findings.append(Finding("G8", "DRIVE_ID_DUP", "index/drives.parquet", count=t.num_rows - nd))
            if "file" in cols:
                idx_files = set(v for v in t["file"].to_pylist() if v)
                data_files = {f for f in files if f.startswith("data/")}
                if idx_files != data_files:
                    findings.append(Finding("G0", "INDEX_FILES", "index/drives.parquet", count=len(idx_files ^ data_files),
                                            detail=f"{len(idx_files - data_files)} indexed files missing, {len(data_files - idx_files)} files not indexed"))
        except Exception as e:                    # noqa: BLE001
            findings.append(Finding("G1", "UNREADABLE", "index/drives.parquet", detail=type(e).__name__))
    return findings


# ------------------------------------------------------------------------------------------------ checksums
def checksum_members(files: list[str]) -> list[str]:
    return [f for f in files if f not in CHECKSUM_EXCLUDE]


def sha256_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def compute_checksums(root: Path, rels: list[str], workers: int = 4) -> dict[str, str]:
    root = Path(root)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        digests = list(ex.map(lambda r: sha256_file(root / r), rels))
    return dict(zip(rels, digests))


def format_checksums(digests: dict[str, str]) -> str:
    """`<sha256>  <relative path>` per line, sorted by path (sha256sum -c compatible)."""
    return "".join(f"{digests[r]}  {r}\n" for r in sorted(digests))


def parse_checksums(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        sha, _, rel = line.partition("  ")
        out[rel.strip()] = sha.strip()
    return out


def verify_checksums(root: Path, files: list[str], workers: int = 4) -> list[Finding]:
    root = Path(root)
    p = root / "CHECKSUMS.sha256"
    if not p.exists():
        return []                                   # reported by G0/MISSING
    listed = parse_checksums(p.read_text())
    actual = compute_checksums(root, checksum_members(files), workers)
    findings = []
    missing = sorted(set(actual) - set(listed))
    extra = sorted(set(listed) - set(actual))
    mismatch = sorted(r for r in actual if r in listed and listed[r] != actual[r])
    for rule, items in (("NOT_LISTED", missing), ("STALE_ENTRY", extra), ("MISMATCH", mismatch)):
        if items:
            findings.append(Finding("CHECKSUMS", rule, "CHECKSUMS.sha256", count=len(items),
                                    detail=gate_mask(", ".join(items[:3]))))
    return findings


# ------------------------------------------------------------------------------------------------ driver
def scan_release(root: Path, private_root: Path | None = None, workers: int = 4, sample_rows: int = 50000,
                 allow: dict | None = None, checksums: bool = True) -> Report:
    """Run every gate over a release tree; the private root (when readable) adds FORBIDDEN and EXACT (G9).
    `sample_rows` bounds the string-value check of drive tables and window shards only (SAMPLED_TABLES); index and
    splits tables are checked over all their distinct values, and the numeric checks always see every row."""
    root = Path(root)
    allow = allow or R.load_allow()
    rep = Report(utc=utc_stamp(), root=str(root))
    if not root.is_dir():
        rep.findings.append(Finding("G0", "MISSING", "", detail="release root is not a directory"))
        return rep
    files = list_release_files(root)
    forbidden: list[str] = []
    private = None
    if R.private_readable(private_root):
        forbidden = PV.forbidden_tokens(Path(private_root))
        if not forbidden:
            rep.warnings.append("forbidden_tokens.txt missing or empty under the private root")
        private = R.private_tokens(Path(private_root))
    else:
        rep.warnings.append("private root not readable: FORBIDDEN and EXACT (G9) rules skipped")
    ctx = Ctx(root, allow, forbidden, private, sample_rows)
    for rel, f, w, size in _run(files, ctx, workers):
        rep.findings += f
        rep.warnings += w
        rep.n_files += 1
        rep.n_bytes += size
    rep.findings += tree_checks(root, files)
    if checksums:
        rep.findings += verify_checksums(root, files, workers)
    rep.findings.sort(key=lambda f: (f.gate, f.rule, f.path, f.column))
    return rep


def write_report(rep: Report, data_root: Path = C.DATA_ROOT, private_root: Path | None = None) -> Path:
    """Masked JSON report: private/logs/gate_<utc>.json when the private root exists (0600), else <data_root>/tmp."""
    private_root = Path(private_root) if private_root else Path(data_root) / "private"
    if private_root.is_dir() and os.access(private_root, os.W_OK):
        out = private_root / "logs" / f"gate_{rep.utc}.json"
        PV.write_json_atomic(rep.to_dict(), out, private=True)
    else:
        out = Path(data_root) / "tmp" / f"gate_{rep.utc}.json"
        PV.write_json_atomic(rep.to_dict(), out)
    return out


def print_summary(rep: Report, report_path: Path | None = None) -> None:
    """Counts only: files, bytes, findings per gate/rule, warnings; never a matched token."""
    print(f"gate: files={rep.n_files} bytes={rep.n_bytes} findings={len(rep.findings)} warnings={len(rep.warnings)} ok={rep.ok}")
    for key, n in rep.counts().items():
        print(f"  {key}: {n}")
    for w in rep.warnings[:10]:
        print(f"  WARN {gate_mask(w)}")
    if report_path is not None:
        print(f"report: {report_path}")
