"""Pre-push gate for the public code repository (GitHub export).

Checks on a git repository directory: forbidden file types among tracked files and in the whole object history
(.ipynb .pyc .parquet .npz .npy .pt .csv .log, except sample/**, vtt/specs_data/*.csv, tests/fixtures/** and the CSV
source data of the manuscript's figures, paper/figure_source/source_data/**/*.csv, whose text is scanned like any file), blobs
over 2 MB, data directories (private / work / release / .cache) as path components, the text rules of
vtt.gate.rules on every tracked text file and every text blob of every commit (`git rev-list --all --objects`),
with NO allow-list for anything under tests/, author / committer e-mails against the private forbidden tokens,
and `import vtt.config` exposing none of the extractor's identity constants. Exit 0 iff no findings, else 2.

Scope of the COORD word rule in the code repository (NAME_ONLY_RULES): the pipeline code must name the private
columns it reads (country, in_house, lat / lon of the route master) and the docs must describe them, so in source
and docs prose the coordinate words are not a leak and the rule is applied to names only: path components, JSON keys,
CSV headers and the column names of tracked parquet files (the synthetic sample). Every identifier rule (HEX16,
HEX12, ROUTE_DATE, ROUTE_NUM, PATHS, VIN, ANON_AUTH, ACRONYM, FORBIDDEN, EXACT) and the soft rules (ISO_DATE, EMAIL)
stay on all text, tests/ included. The release gate (vtt.gate.scan) is unchanged: COORD applies to all its text.

PDF files (tracked and in history; recognised by the %PDF- header at the start of the file, or anywhere in the first
KB of a file named *.pdf) are scanned through five views together: the page text (pdftotext), the document
information including custom keys (pdfinfo -custom), the XMP metadata packet of the catalog (pdfinfo -meta), the raw
file bytes with every stream body removed (uncompressed object dictionaries, trailers, bytes after %%EOF and incremental
updates), and the inflated bodies of the streams that hold objects or metadata rather than drawings: object streams
(/Type /ObjStm: link annotations and URIs, outlines, the info dictionaries pdfTeX copies from included figures),
/Type /Metadata streams at any level and /Type /EmbeddedFile attachments. Page content and image streams are left
out; their compressed bytes otherwise match the e-mail and VIN patterns at random. Fail closed: a PDF the tools cannot
read, any PDF when poppler is missing, an encrypted PDF (/Encrypt), and one of those object or metadata streams that is
not plain FlateDecode or does not inflate are findings (PDF_UNREADABLE, PDF_ENCRYPTED). A PDF without any page text
is reported as a warning so a person looks at it. The CSV exemption under paper/figure_source/source_data/ requires valid UTF-8 text
(anything else is an UNREADABLE finding). Geodata files (.geojson .gpx .kml) are forbidden types except the pinned
Natural Earth outline of the manuscript's map figure. Suffix checks ignore case. The CSV exemption and the PDF views were
approved by the repository owner on 16 Nov 2026.
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq

from vtt import config as C
from vtt import private as PV
from vtt.gate import rules as R
from vtt.gate import scan as SC
from vtt.gate.rules import Finding, Report, gate_mask

FORBIDDEN_SUFFIXES = (".ipynb", ".pyc", ".parquet", ".npz", ".npy", ".pt", ".csv", ".log", ".geojson", ".gpx", ".kml")
EXEMPT_PREFIXES = ("sample/", "tests/fixtures/")
EXEMPT_CSV_DIR = "vtt/specs_data"
EXEMPT_CSV_PREFIXES = ("paper/figure_source/source_data/",)    # the plotted data behind the manuscript's figures
# the one geodata file: Natural Earth 1:110m admin-0 outlines (public domain) behind the map figure, pinned by content
PINNED_FILES = {"paper/figure_source/source_data/teaser_C/ne_110m_admin0_countries.geojson":
                "6866c877d39cba9c357620878839b336d569f8c662d3cfab4cb1dbe2d39c977f"}
MAX_BLOB_BYTES = 2 * 1024 * 1024
BAD_DIR_COMPONENTS = frozenset({"private", "work", "release", ".cache"})
IDENTITY_CONST_NAMES = ("MISFILED_DONGLE_DIRS", "LAB_DONGLES", "T0_EXTRA_IDENTITIES", "CONFLICTING_DUPLICATE_CLIPS")
NOREPLY_SUFFIX = "@users.noreply.github.com"
TEXT_PROBE = 8192
NAME_ONLY_RULES = ("COORD",)       # word rules applied to names (paths, JSON keys, CSV headers, parquet columns), not to prose


def git(repo: Path, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return r.stdout if binary else r.stdout.decode(errors="replace")


def tracked_files(repo: Path) -> list[str]:
    out = git(repo, "ls-files", "-z")
    return sorted(p for p in out.split("\0") if p)


def exempt_csv(path: str) -> bool:
    """A figure-source CSV (exempt from the type rule, but it must be readable UTF-8 text)."""
    return path.startswith(EXEMPT_CSV_PREFIXES) and path.lower().endswith(".csv")


def is_exempt(path: str) -> bool:
    low = path.lower()
    return (path.startswith(EXEMPT_PREFIXES) or (posixpath.dirname(path) == EXEMPT_CSV_DIR and low.endswith(".csv"))
            or exempt_csv(path) or path in PINNED_FILES)


def forbidden_type(path: str) -> bool:
    return path.lower().endswith(FORBIDDEN_SUFFIXES) and not is_exempt(path)


def pinned_mismatch(path: str, data: bytes) -> bool:
    """A pinned file whose content is not the pinned content (the exemption covers exactly one version)."""
    import hashlib
    return path in PINNED_FILES and hashlib.sha256(data).hexdigest() != PINNED_FILES[path]


def bad_dir(path: str) -> bool:
    """A data directory name among the DIRECTORY components (vtt/private.py is a module, not a data dir)."""
    return any(part in BAD_DIR_COMPONENTS for part in PurePosixPath(path).parts[:-1])


def is_text(data: bytes) -> bool:
    return b"\0" not in data[:TEXT_PROBE]


UNREADABLE_PDF = "\x00PDF_UNREADABLE"          # sentinel returned by scan_text (never valid text)
ENCRYPTED_PDF = "\x00PDF_ENCRYPTED"
UNREADABLE_TEXT = "\x00UNREADABLE"
STREAM_BODY = re.compile(rb"stream\r?\n.*?endstream", re.S)


def is_pdf(data: bytes, path: str = "") -> bool:
    """%PDF- at the start of the file (after optional whitespace / BOM); for *.pdf names anywhere in the first KB."""
    head = data[:1024]
    if head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"%PDF-"):
        return True
    return path.lower().endswith(".pdf") and b"%PDF-" in head


STREAM_OBJ = re.compile(rb"<<(.{0,4000}?)>>\s*stream\r?\n", re.S)
KEEP_STREAM = re.compile(rb"/Type\s*/(ObjStm|Metadata|EmbeddedFile)\b|/Subtype\s*/XML\b")


def inflated_object_streams(data: bytes) -> str | None:
    """Inflated bodies of the object / metadata / attachment streams, or None if one of them cannot be inflated."""
    import zlib
    out = []
    for m in STREAM_OBJ.finditer(data):
        head = m.group(1)
        if not KEEP_STREAM.search(head):
            continue
        end = data.find(b"endstream", m.end())
        if end < 0:
            return None
        body = data[m.end():end]
        filt = re.findall(rb"/Filter\s*(\[[^\]]*\]|/[A-Za-z0-9]+)", head)
        names = re.findall(rb"/([A-Za-z0-9]+)", filt[0]) if filt else []
        if not names:
            out.append(body)
        elif names == [b"FlateDecode"] and b"/DecodeParms" not in head:
            try:
                out.append(zlib.decompressobj().decompress(body))
            except zlib.error:
                return None
        else:
            return None
    return b"\n".join(out).decode(errors="replace")


def pdf_text(data: bytes) -> str | None:
    """Page text, document information (custom keys included), XMP packet and the raw bytes without stream bodies of a
    PDF; None if poppler is missing or cannot read the file (the caller turns None into a finding)."""
    import shutil
    import tempfile
    if not (shutil.which("pdftotext") and shutil.which("pdfinfo")):
        return None
    with tempfile.TemporaryDirectory(prefix="vtt_gate_pdf_", dir=os.environ.get("TMPDIR") or None) as d:
        f = Path(d) / "doc.pdf"
        f.write_bytes(data)
        try:
            body = subprocess.run(["pdftotext", "-q", "-enc", "UTF-8", str(f), "-"], capture_output=True, check=True,
                                  timeout=300).stdout
            info = subprocess.run(["pdfinfo", "-enc", "UTF-8", "-custom", str(f)], capture_output=True, check=True,
                                  timeout=120).stdout
            meta = subprocess.run(["pdfinfo", "-enc", "UTF-8", "-meta", str(f)], capture_output=True, check=True,
                                  timeout=120).stdout
        except (subprocess.SubprocessError, OSError):
            return None
    skeleton = STREAM_BODY.sub(b" ", data)
    inner = inflated_object_streams(data)
    if inner is None:
        return None
    return VIEW_SEP.join([*(x.decode(errors="replace") for x in (body, info, meta, skeleton)), inner])


VIEW_SEP = "\n\x00VIEW\n"                       # separates the four PDF views (never matched by any rule)


def pdf_has_page_text(text: str) -> bool:
    return bool(text.split(VIEW_SEP, 1)[0].strip()) if text else False


def scan_text(path: str, data: bytes) -> str | None:
    """The text the rules see for one file / blob: the four PDF views of a PDF (UNREADABLE_PDF if they cannot be read),
    else the text of a text file; a figure-source CSV that is not UTF-8 text gives UNREADABLE_TEXT; None for binaries."""
    if is_pdf(data, path):
        if re.search(rb"/Encrypt\b", data):
            return ENCRYPTED_PDF
        t = pdf_text(data)
        return UNREADABLE_PDF if t is None else t
    if exempt_csv(path):
        try:
            if not is_text(data):
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "NUL byte")
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return UNREADABLE_TEXT
    return data.decode(errors="replace") if is_text(data) else None


def scan_view(path: str, text: str, allow: dict, forbidden, private, prefix: str = "") -> list[Finding]:
    """Rules on one scan_text view, with the unreadable sentinels turned into findings."""
    if text == UNREADABLE_PDF:
        return [Finding("PRE_PUSH", prefix + "PDF_UNREADABLE", gate_mask(path))]
    if text == ENCRYPTED_PDF:
        return [Finding("PRE_PUSH", prefix + "PDF_ENCRYPTED", gate_mask(path), detail="decrypt or rewrite the PDF before commit")]
    if text == UNREADABLE_TEXT:
        return [Finding("PRE_PUSH", prefix + "UNREADABLE", gate_mask(path), detail="figure-source CSV is not UTF-8 text")]
    return scan_source(path, text, allow, forbidden, private)


def csv_header(text: str) -> list[str]:
    first = text.splitlines()[0] if text.strip() else ""
    return [c.strip().strip('"').strip("'") for c in first.split(",") if c.strip()]


def scan_source(path: str, text: str, allow: dict, forbidden, private) -> list[Finding]:
    """Text rules on one repository file; under tests/ nothing is allow-listed (ISO dates and e-mails are hard).
    NAME_ONLY_RULES are dropped from the prose hits and applied to JSON keys and CSV headers instead."""
    in_tests = path.startswith("tests/")
    iso_allowed = (not in_tests) and path in allow.get("pre_push_iso_date_files", [])
    allowed_email = "" if in_tests else allow.get("email", "")
    readme = PurePosixPath(path).name.startswith("README")
    hits = R.scan_text(text, is_json=path.endswith(".json"), readme=readme, allowed_email=allowed_email,
                       iso_allowed=iso_allowed, forbidden=forbidden, private=private)
    for rule in NAME_ONLY_RULES:
        hits.pop(rule, None)
    findings = [Finding("PRE_PUSH", rule, gate_mask(path), count=n) for rule, n in hits.items()]
    if path.endswith(".json"):
        for rule, n in R.name_rule_hits(R.json_keys(text)).items():
            findings.append(Finding("PRE_PUSH", rule, gate_mask(path), column="<json_keys>", count=n))
    if path.endswith(".csv"):
        for rule, n in R.name_rule_hits(csv_header(text)).items():
            findings.append(Finding("PRE_PUSH", rule, gate_mask(path), column="<columns>", count=n))
    return findings


def scan_path_names(path: str) -> list[Finding]:
    """The coordinate-column rule on the path components of a tracked file (a file or folder named like one)."""
    comps = [PurePosixPath(c).stem for c in PurePosixPath(path).parts]
    n = sum(1 for c in comps if R.col_coord(c))
    return [Finding("PRE_PUSH", "COORD", gate_mask(path), column="<path>", count=n)] if n else []


def scan_parquet_names(repo: Path, path: str, forbidden, private) -> list[Finding]:
    """Column names of a tracked parquet file (the exempt synthetic sample) against the name and text rules."""
    try:
        names = list(pq.read_schema(repo / path).names)
    except Exception as e:                            # noqa: BLE001
        return [Finding("PRE_PUSH", "UNREADABLE", gate_mask(path), detail=type(e).__name__)]
    return [Finding("PRE_PUSH", rule, gate_mask(path), column="<columns>", count=n)
            for rule, n in R.scan_names(names, forbidden=forbidden, private=private).items()]


def history_blobs(repo: Path) -> dict[str, set[str]]:
    """blob sha -> set of paths it was ever stored under (all refs)."""
    out: dict[str, set[str]] = {}
    for line in git(repo, "rev-list", "--all", "--objects").splitlines():
        sha, _, path = line.partition(" ")
        if path:
            out.setdefault(sha, set()).add(path)
    types = {}
    proc = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-check"], input="\n".join(out).encode(),
                          capture_output=True, check=True)
    for line in proc.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 3:
            types[parts[0]] = (parts[1], int(parts[2]))
    return {sha: paths for sha, paths in out.items() if types.get(sha, ("", 0))[0] == "blob"}


def blob_contents(repo: Path, shas: list[str]):
    """Yield (sha, bytes) for the given blobs through one `git cat-file --batch`."""
    if not shas:
        return
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    data, _ = proc.communicate(("\n".join(shas) + "\n").encode())
    pos = 0
    while pos < len(data):
        nl = data.index(b"\n", pos)
        header = data[pos:nl].decode(errors="replace").split()
        pos = nl + 1
        if len(header) < 3:
            continue
        sha, size = header[0], int(header[2])
        yield sha, data[pos:pos + size]
        pos += size + 1


def accepted_history(allow: dict) -> dict[tuple[str, str], str]:
    """(blob sha, rule) -> reason for findings in already-published history that were reviewed and found harmless
    (gate_allow.json "history_accepted"). A pushed commit cannot be fixed without rewriting public history, so the
    acceptance is bound to the immutable blob content: a blob with any other content, under any path, is a new finding.
    An entry needs a 40-hex blob sha, a rule name and a non-empty reason; anything else is ignored. Blobs that were
    ever stored under tests/ are never accepted (the tests directory has no allow-list). The sha is written in
    gate_allow.json with dashes between 8-hex groups so that the allow-list itself does not look like an identifier to
    the hex rules; dashes and spaces are removed here."""
    out: dict[tuple[str, str], str] = {}
    for e in allow.get("history_accepted", []) or []:
        sha = str(e.get("blob", "")).lower().replace("-", "").replace(" ", "")
        rule, reason = str(e.get("rule", "")), str(e.get("reason", "")).strip()
        if len(sha) == 40 and all(c in "0123456789" + "abcdef" for c in sha) and rule and reason:   # no 16-hex literal here
            out[(sha, rule)] = reason
    return out


def check_history(repo: Path, allow: dict, forbidden, private) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    warnings: list[str] = []
    accepted = accepted_history(allow)
    blobs = history_blobs(repo)
    shas = sorted(blobs)
    sizes = {}
    proc = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-check"], input="\n".join(shas).encode(),
                          capture_output=True, check=True)
    for line in proc.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 3:
            sizes[parts[0]] = int(parts[2])
    for sha in shas:
        paths = sorted(blobs[sha])
        shown = gate_mask(paths[0])
        if sizes.get(sha, 0) > MAX_BLOB_BYTES:
            findings.append(Finding("PRE_PUSH", "BLOB_SIZE", shown, detail=f"{sizes[sha]} bytes in history"))
        if any(forbidden_type(p) for p in paths):
            findings.append(Finding("PRE_PUSH", "HISTORY_TYPE", shown))
    for sha, data in blob_contents(repo, shas):
        paths = sorted(blobs[sha])
        path = next((p for p in paths if p.startswith("tests/")), paths[0])   # strictest view of a shared blob
        text = scan_text(path, data)
        if text is None:
            continue
        if pinned_mismatch(path, data):
            findings.append(Finding("PRE_PUSH", "HISTORY_TYPE", gate_mask(path), detail="pinned file changed"))
        for f in scan_view(path, text, allow, forbidden, private):
            reason = accepted.get((sha, f.rule))
            if reason is not None and not any(p.startswith("tests/") for p in paths):
                warnings.append(f"accepted history finding {f.rule} in {gate_mask(path)} (blob {sha[:12]}): {reason}")
                continue
            f.rule = "HISTORY_" + f.rule
            findings.append(f)
    for path in sorted({p for paths in blobs.values() for p in paths}):
        for f in scan_path_names(path):
            f.rule = "HISTORY_" + f.rule
            findings.append(f)
    return findings, warnings


def check_authors(repo: Path, forbidden) -> tuple[list[Finding], list[str]]:
    findings, warnings = [], []
    emails = sorted(set(e.strip() for e in git(repo, "log", "--all", "--format=%ae%n%ce").splitlines() if e.strip()))
    for e in emails:
        if forbidden and R.hits_forbidden(e, forbidden):
            findings.append(Finding("PRE_PUSH", "AUTHOR_EMAIL", detail=gate_mask(e)))
        if not e.lower().endswith(NOREPLY_SUFFIX):
            warnings.append(f"author e-mail is not a GitHub noreply address: {gate_mask(e)}")
    return findings, warnings


def check_config_exposure(repo: Path, python: str = sys.executable) -> tuple[list[Finding], list[str]]:
    """`import vtt.config` in a subprocess rooted at the repo must expose none of IDENTITY_CONST_NAMES; the source text
    of vtt/config.py must not mention them either."""
    findings, warnings = [], []
    cfg = repo / "vtt" / "config.py"
    if not cfg.exists():
        return findings, ["vtt/config.py not present in the repository"]
    text = cfg.read_text(errors="replace")
    mentioned = [n for n in IDENTITY_CONST_NAMES if n in text]
    if mentioned:
        findings.append(Finding("PRE_PUSH", "CONFIG_TEXT", "vtt/config.py", count=len(mentioned), detail=", ".join(mentioned)))
    code = "import json, vtt.config as c; print(json.dumps(sorted(n for n in dir(c) if not n.startswith('_'))))"
    env = dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1")
    try:
        r = subprocess.run([python, "-c", code], cwd=str(repo), env=env, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return findings, [f"import vtt.config could not be run: {type(e).__name__}"]
    if r.returncode != 0:
        return findings, ["import vtt.config failed in the repository: " + gate_mask(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")]
    exposed = [n for n in IDENTITY_CONST_NAMES if n in set(json.loads(r.stdout))]
    if exposed:
        findings.append(Finding("PRE_PUSH", "CONFIG_EXPOSES", "vtt/config.py", count=len(exposed), detail=", ".join(exposed)))
    return findings, warnings


def run(repo: Path, private_root: Path | None = None, allow: dict | None = None, python: str = sys.executable) -> Report:
    repo = Path(repo)
    allow = allow or R.load_allow()
    rep = Report(utc=SC.utc_stamp(), root=str(repo))
    try:
        git(repo, "rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, FileNotFoundError):
        rep.findings.append(Finding("PRE_PUSH", "NOT_A_REPO", gate_mask(str(repo))))
        return rep
    forbidden: list[str] = []
    private = None
    if R.private_readable(private_root):
        forbidden = PV.forbidden_tokens(Path(private_root))
        if not forbidden:
            rep.warnings.append("forbidden_tokens.txt missing or empty under the private root")
        private = R.private_tokens(Path(private_root))
    else:
        rep.warnings.append("private root not readable: FORBIDDEN / EXACT rules and author e-mail check skipped")
    files = tracked_files(repo)
    for path in files:
        if forbidden_type(path):
            rep.findings.append(Finding("PRE_PUSH", "FILE_TYPE", gate_mask(path)))
        if bad_dir(path):
            rep.findings.append(Finding("PRE_PUSH", "DATA_DIR", gate_mask(path)))
        rep.findings += scan_path_names(path)
        p = repo / path
        if not p.is_file():
            continue
        size = p.stat().st_size
        if size > MAX_BLOB_BYTES:
            rep.findings.append(Finding("PRE_PUSH", "FILE_SIZE", gate_mask(path), detail=f"{size} bytes"))
        data = p.read_bytes()
        if pinned_mismatch(path, data):
            rep.findings.append(Finding("PRE_PUSH", "FILE_TYPE", gate_mask(path), detail="pinned file changed"))
        text = scan_text(path, data)
        if text is not None:
            rep.findings += scan_view(path, text, allow, forbidden, private)
            if is_pdf(data, path) and text not in (UNREADABLE_PDF, ENCRYPTED_PDF) and not pdf_has_page_text(text):
                rep.warnings.append(f"PDF without page text (outlined or image-only), look at it: {gate_mask(path)}")
        elif path.lower().endswith(".parquet"):
            rep.findings += scan_parquet_names(repo, path, forbidden, private)
        rep.n_files += 1
        rep.n_bytes += size
    f, w = check_history(repo, allow, forbidden, private)
    rep.findings += f
    rep.warnings += w
    f, w = check_authors(repo, forbidden)
    rep.findings += f
    rep.warnings += w
    f, w = check_config_exposure(repo, python)
    rep.findings += f
    rep.warnings += w
    rep.findings.sort(key=lambda x: (x.gate, x.rule, x.path, x.column))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VTT pre-push gate for a git repository (exit 0 iff no findings)")
    ap.add_argument("repo", type=Path)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--private-root", type=Path, default=None, help="default <data-root>/private")
    ap.add_argument("--python", default=sys.executable, help="interpreter used for the `import vtt.config` probe")
    a = ap.parse_args(argv)
    private_root = a.private_root or (a.data_root / "private")
    rep = run(a.repo, private_root=private_root, python=a.python)
    path = SC.write_report(rep, data_root=a.data_root, private_root=private_root)
    SC.print_summary(rep, path)
    return 0 if rep.ok else 2


if __name__ == "__main__":
    sys.exit(main())
