"""Pre-push gate for the public code repository (GitHub export).

Checks on a git repository directory: forbidden file types among tracked files and in the whole object history
(.ipynb .pyc .parquet .npz .npy .pt .csv .log, except sample/**, vtt/specs_data/*.csv and tests/fixtures/**), blobs
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
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq

from vtt import config as C
from vtt import private as PV
from vtt.gate import rules as R
from vtt.gate import scan as SC
from vtt.gate.rules import Finding, Report, gate_mask

FORBIDDEN_SUFFIXES = (".ipynb", ".pyc", ".parquet", ".npz", ".npy", ".pt", ".csv", ".log")
EXEMPT_PREFIXES = ("sample/", "tests/fixtures/")
EXEMPT_CSV_DIR = "vtt/specs_data"
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


def is_exempt(path: str) -> bool:
    return path.startswith(EXEMPT_PREFIXES) or (posixpath.dirname(path) == EXEMPT_CSV_DIR and path.endswith(".csv"))


def forbidden_type(path: str) -> bool:
    return path.endswith(FORBIDDEN_SUFFIXES) and not is_exempt(path)


def bad_dir(path: str) -> bool:
    """A data directory name among the DIRECTORY components (vtt/private.py is a module, not a data dir)."""
    return any(part in BAD_DIR_COMPONENTS for part in PurePosixPath(path).parts[:-1])


def is_text(data: bytes) -> bool:
    return b"\0" not in data[:TEXT_PROBE]


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


def check_history(repo: Path, allow: dict, forbidden, private) -> list[Finding]:
    findings: list[Finding] = []
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
        if not is_text(data):
            continue
        paths = sorted(blobs[sha])
        path = next((p for p in paths if p.startswith("tests/")), paths[0])   # strictest view of a shared blob
        for f in scan_source(path, data.decode(errors="replace"), allow, forbidden, private):
            f.rule = "HISTORY_" + f.rule
            findings.append(f)
    for path in sorted({p for paths in blobs.values() for p in paths}):
        for f in scan_path_names(path):
            f.rule = "HISTORY_" + f.rule
            findings.append(f)
    return findings


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
        if is_text(data):
            rep.findings += scan_source(path, data.decode(errors="replace"), allow, forbidden, private)
        elif path.endswith(".parquet"):
            rep.findings += scan_parquet_names(repo, path, forbidden, private)
        rep.n_files += 1
        rep.n_bytes += size
    rep.findings += check_history(repo, allow, forbidden, private)
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
