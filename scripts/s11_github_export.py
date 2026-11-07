#!/usr/bin/env python3
"""S11: assemble the public code repository under <data-root>/github_export and prepare the push.

Steps: copy vtt/, scripts/, starter_kit/, experiments/, tests/, docs/, README.md, LICENSE, THIRD_PARTY_NOTICES.md,
requirements.txt, pyproject.toml, .gitignore, without caches / byte-code and without any path that the repository's own
.gitignore excludes (decided by `git check-ignore`, so the export holds exactly what `git add -A` would track); rewrite
local absolute paths in every text file to environment-variable placeholders ($R, $VTT_DATA_ROOT, $VTT_RAW_ROOT, ...;
vtt/config.py included unless --keep-config-paths, because the code gate has no path exemption; file modes are kept);
write sample/ (synthetic drives from vtt.synth when present, else vtt.synth_sample; at most 500 KB); pre-commit text scan
with the gate rules on the files git would track (exit 2 on any hit, nothing is committed); fresh `git init -b main` with
repository-local user.name / user.email (a GitHub noreply address); one commit; a tracked-files check (every file of the
export must be in the commit, so a .gitignore rule can never drop part of the sample silently); the code gate
vtt.gate.pre_push (exit 2 on findings); finally the exact `ssh -T`, `git remote add` and `git push` commands are printed
for the user (this script never pushes).
Idempotent: the marker .vtt_export.json is written before anything else, so an export of this script (complete or
refused) is replaced by the next run; anything else at --out needs --force. Exit codes: 0 ok, 2 refused / failed check.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from vtt import config as C
from vtt.private import mask

EXPORT_ITEMS = ("vtt", "scripts", "starter_kit", "experiments", "tests", "docs", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
                "requirements.txt", "pyproject.toml", ".gitignore")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".ipynb"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".sh", ".json", ".yaml", ".yml", ".csv", ""}
MARKER = ".vtt_export.json"
SAMPLE_MAX_BYTES = 500_000
NOREPLY = "@users.noreply.github.com"
# ordered: longest / most specific first; every value is a shell-style placeholder documented in README.md (Environment)
PATH_REWRITES: tuple[tuple[str, str], ...] = (
    (str(C.TOCF_ROOT), "$VTT_TOCF_ROOT"),
    (str(C.REUSE_EXISTING_SIGNALS.parent.parent), "$VTT_TOCF_DATA"),
    (str(C.TAXONOMY_DIR), "$VTT_TAXONOMY_DIR"),
    (str(C.ROUTE_MASTER_CSV), "$VTT_ROUTE_MASTER"),
    (str(C.ROUTE_MASTER_CSV.parent.parent), "$VTT_OVERVIEW_DIR"),
    (str(C.RAW_ROOT), "$VTT_RAW_ROOT"),
    (str(C.OPENDBC_LOCAL_FALLBACK), "$VTT_OPENDBC_LOCAL"),
    (C.VENV_PY, "$PY"),
    (C.CONDA_PY, "$CPY"),
    (str(C.DATA_ROOT), "$VTT_DATA_ROOT"),
    (str(Path(__file__).resolve().parents[1]), "$R"),
    (str(Path(__file__).resolve().parents[2]), "$VTT_PROJECTS_DIR"),
    (str(C.DATA_ROOT.parent), "$VTT_DATA_PARENT"),
    (str(C.RAW_ROOT.parent.parent), "$VTT_DATASETS_DIR"),
    (str(Path.home()), "$HOME"),
)
PLACEHOLDERS = tuple(dict.fromkeys(dst for _, dst in PATH_REWRITES))      # every placeholder the rewrite can emit (README table)
PLAN_REF = re.compile(r"`?~/\.claude/plans/[^\s`)]+`?")
# fallback leftover-path rule (gate module absent): absolute prefixes only, like vtt.gate.rules.PATHS
LEFTOVER = re.compile(r"(?<![A-Za-z0-9_$}>.])/(?:home|data|mnt|tmp)/")


# ------------------------------------------------------------------------------------------------ selection
def rewrite_text(text: str) -> tuple[str, int]:
    """Local absolute paths -> placeholders; internal plan references -> a neutral note. Returns (text, n_changes)."""
    n = 0
    for src, dst in PATH_REWRITES:
        if src and src in text:
            n += text.count(src)
            text = text.replace(src, dst)
    text, k = PLAN_REF.subn("(internal design plan, not published)", text)
    return text, n + k


def _skip(path: Path) -> bool:
    return any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts) or path.suffix in SKIP_SUFFIXES


def git_ignored(gitignore_text: str, rel_paths) -> set[str]:
    """The relative (posix) paths among `rel_paths` that the given .gitignore text excludes, decided by git itself:
    `git check-ignore --stdin -z` in a throw-away repository holding only that ignore file (paths need not exist).
    Raises RuntimeError when git is unusable (the export needs git anyway)."""
    paths = [p for p in rel_paths if p and p != ".gitignore"]
    if not paths or not gitignore_text.strip():
        return set()
    try:
        with tempfile.TemporaryDirectory(prefix="vtt_ignore_") as tmp:
            subprocess.run(["git", "-C", tmp, "init", "-q"], check=True, capture_output=True)
            Path(tmp, ".gitignore").write_text(gitignore_text)
            r = subprocess.run(["git", "-C", tmp, "check-ignore", "--stdin", "-z"], input="\0".join(paths).encode(),
                               capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"git is required to evaluate .gitignore ({type(exc).__name__})") from exc
    if r.returncode not in (0, 1):                     # 0 = some ignored, 1 = none, anything else = error
        raise RuntimeError("git check-ignore failed: " + mask(r.stderr.decode(errors="replace").strip()[:200]))
    return {p for p in r.stdout.decode(errors="replace").split("\0") if p}


def candidate_files(repo_root: Path) -> tuple[list[Path], list[str]]:
    """Relative files of EXPORT_ITEMS after the cache / byte-code skip, plus the missing items."""
    files, missing = [], []
    for item in EXPORT_ITEMS:
        src = repo_root / item
        if not src.exists():
            missing.append(item)
            continue
        for p in ([src] if src.is_file() else sorted(q for q in src.rglob("*") if q.is_file())):
            rel = p.relative_to(repo_root)
            if not _skip(rel):
                files.append(rel)
    return files, missing


def select_files(repo_root: Path) -> dict:
    """{'files': [relative Path], 'ignored': n, 'missing_items': [...]}: the candidates minus what .gitignore excludes."""
    files, missing = candidate_files(repo_root)
    gi = repo_root / ".gitignore"
    ignored = git_ignored(gi.read_text(errors="replace"), [f.as_posix() for f in files]) if gi.exists() else set()
    return {"files": [f for f in files if f.as_posix() not in ignored], "ignored": len(ignored), "missing_items": missing}


def export_files(out: Path) -> tuple[list[str], list[str]]:
    """(tracked, ignored): relative posix paths of the files of an export that git would track, and those on disk that
    the export's own .gitignore excludes (marker and .git/ left out of both)."""
    rels = [p.relative_to(out).as_posix() for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != MARKER and not _skip(p.relative_to(out))]
    gi = out / ".gitignore"
    ignored = git_ignored(gi.read_text(errors="replace"), rels) if gi.exists() else set()
    return [r for r in rels if r not in ignored], sorted(ignored)


# ------------------------------------------------------------------------------------------------ copy + sample
def copy_tree(repo_root: Path, out: Path, keep_config_paths: bool = False) -> dict:
    """Copy the selected files into `out`, rewriting text files (modes kept); returns counts."""
    sel = select_files(repo_root)
    counts = {"files": 0, "rewritten": 0, "replacements": 0, "ignored": sel["ignored"], "missing_items": sel["missing_items"]}
    for rel in sel["files"]:
        p = repo_root / rel
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        is_text = p.suffix in TEXT_SUFFIXES and b"\0" not in p.read_bytes()[:8192]
        if is_text and not (keep_config_paths and rel.as_posix() == "vtt/config.py"):
            text, n = rewrite_text(p.read_text(errors="replace"))
            dst.write_text(text)
            shutil.copymode(p, dst)                    # executable bits of scripts survive the rewrite
            if n:
                counts["rewritten"] += 1
                counts["replacements"] += n
        else:
            shutil.copy2(p, dst)
        counts["files"] += 1
    return counts


def write_sample(out: Path) -> dict:
    """sample/ from vtt.synth.write_sample when that module offers it, else vtt.synth_sample; the result carries
    `too_big` (> SAMPLE_MAX_BYTES) for the caller to refuse."""
    mod = None
    try:
        cand = importlib.import_module("vtt.synth")
        mod = cand if hasattr(cand, "write_sample") else None
    except ImportError:
        mod = None
    if mod is None:
        mod = importlib.import_module("vtt.synth_sample")
    res = mod.write_sample(out / "sample", n_vehicles=2, n_drives=2)
    res["source"] = mod.__name__
    res["too_big"] = res.get("bytes", 0) > SAMPLE_MAX_BYTES
    return res


# ------------------------------------------------------------------------------------------------ pre-commit scan
def scan_text_file(rel: str, text: str) -> dict[str, int]:
    """Text rules of the code gate on one file: `vtt.gate.pre_push.scan_source` (tests/ without allow-list, name-only
    rules applied to JSON keys / CSV headers, not prose) when importable, else `vtt.gate.rules.scan_text`, else the
    identifier regexes of vtt.private plus the leftover-path rule of this script (the gate's PATHS rule covers it)."""
    hits: dict[str, int] = {}
    try:
        from vtt.gate import pre_push, rules as R
        for f in pre_push.scan_source(rel, text, R.load_allow(), forbidden=[], private=None):
            hits[f.rule] = hits.get(f.rule, 0) + int(f.count)
        return hits                                        # the gate's PATHS rule already covers leftover local paths
    except ImportError:
        try:
            from vtt.gate import rules as R
            allow = R.load_allow()
            in_tests = rel.startswith("tests/")
            hits = R.scan_text(text, is_json=rel.endswith(".json"), readme=Path(rel).name.startswith("README"),
                               allowed_email="" if in_tests else allow.get("email", ""),
                               iso_allowed=(not in_tests) and rel in allow.get("pre_push_iso_date_files", []))
        except ImportError:
            from vtt import private as PV
            hits = {k: v for k, v in {"HEX16": len(PV.HEX16.findall(text)), "ROUTE_DATE": len(PV.ROUTE_DATE.findall(text)),
                                      "ROUTE_NUM": len(PV.ROUTE_NUM.findall(text))}.items() if v}
    n_left = len(LEFTOVER.findall(text))
    if n_left:
        hits["LOCAL_PATH"] = n_left
    return hits


def precheck(out: Path) -> list[tuple[str, str, int]]:
    """(masked path, rule, hits) for every text file git would track in the export; empty = clean."""
    findings = []
    tracked, _ = export_files(out)
    for rel in tracked:
        data = (out / rel).read_bytes()
        if b"\0" in data[:8192]:
            continue
        for rule, n in scan_text_file(rel, data.decode(errors="replace")).items():
            findings.append((mask(rel), rule, int(n)))
    return findings


# ------------------------------------------------------------------------------------------------ git + gate
def git(out: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(out), *args], check=True, capture_output=True, text=True).stdout.strip()


def git_init_commit(out: Path, author_name: str, author_email: str, message: str) -> str:
    """Fresh repository on `main` with repository-local identity; returns the commit sha."""
    git(out, "init", "-q", "-b", "main")
    git(out, "config", "user.name", author_name)
    git(out, "config", "user.email", author_email)
    exclude = out / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text(f"{MARKER}\n")
    git(out, "add", "-A")
    git(out, "commit", "-q", "-m", message)
    return git(out, "rev-parse", "HEAD")


def tracked_mismatch(out: Path) -> tuple[int, int, int]:
    """(n_tracked, n_differing, n_ignored_on_disk) between the commit and the export directory; (n, 0, 0) is the only
    acceptable outcome (a file on disk that .gitignore drops would silently miss the push)."""
    tracked = {p for p in git(out, "ls-files", "-z").split("\0") if p}
    want, ignored = export_files(out)
    return len(tracked), len(set(want) ^ tracked), len(ignored)


def run_gate(out: Path, data_root: Path, private_root: Path | None, gate_fn=None) -> tuple[bool, list[str]]:
    """The code gate on the committed export: `gate_fn(out) -> (ok, lines)` when injected, else
    vtt.gate.pre_push.run (imported lazily; a missing gate module is a loud failure, never a silent pass)."""
    if gate_fn is not None:
        return gate_fn(out)
    try:
        from vtt.gate import pre_push
        from vtt.gate import scan as SC
    except ImportError as exc:
        return False, [f"GATE MODULE MISSING ({type(exc).__name__}): vtt/gate/pre_push.py is required before any push"]
    rep = pre_push.run(out, private_root=private_root, python=sys.executable)
    lines = [f"gate: files={rep.n_files} findings={len(rep.findings)} warnings={len(rep.warnings)} ok={rep.ok}"]
    lines += [f"  {k}: {n}" for k, n in rep.counts().items()]
    lines += [f"  WARN {mask(w)}" for w in rep.warnings[:10]]
    try:
        lines.append(f"report: {SC.write_report(rep, data_root=data_root, private_root=private_root)}")
    except Exception as exc:                                # noqa: BLE001 - the report file is a convenience
        lines.append(f"report not written: {type(exc).__name__}")
    return rep.ok, lines


def github_ssh_aliases() -> list[str]:
    """Host aliases of ~/.ssh/config whose HostName is github.com (plus 'github.com' itself)."""
    cfg = Path.home() / ".ssh" / "config"
    aliases, current = [], []
    if cfg.exists():
        for line in cfg.read_text(errors="replace").splitlines():
            s = line.strip()
            if s.lower().startswith("host "):
                current = s.split()[1:]
            elif s.lower().startswith("hostname") and "github.com" in s.lower():
                aliases += [h for h in current if "*" not in h]
    return sorted(set(aliases) | {"github.com"})


def push_instructions(out: Path, aliases: list[str]) -> str:
    repo = C.GITHUB_REPO
    lines = ["# 1. find the alias that authenticates as the repository owner (the reply names the account):"]
    lines += [f"ssh -T git@{a}" for a in aliases]
    lines += ["# 2. add the remote (the repository name starts with '-': keep the quotes) and push:",
              f"cd '{out}'", f"git remote add origin 'git@<alias>:{repo}.git'", "git push -u origin main",
              "# 3. verify the published tree (fresh clone under $TMPDIR, code gate + tests):",
              f"git clone 'git@<alias>:{repo}.git' \"$TMPDIR/vtt_clone\"",
              "python scripts/s09b_gate.py --pre-push \"$TMPDIR/vtt_clone\"",
              "(cd \"$TMPDIR/vtt_clone\" && python -m pytest -q)"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ driver
def write_marker(out: Path, **fields) -> None:
    (out / MARKER).write_text(json.dumps({"tool": "s11_github_export", **fields}, indent=1))


def prepare_out(out: Path, force: bool) -> str | None:
    """Remove a previous export of this script (marker present: complete or refused) or anything when --force; returns
    the refusal message otherwise (None = ready)."""
    if not out.exists():
        return None
    if any(out.iterdir()) and not force and not (out / MARKER).exists():
        return f"{out} is not empty and not a previous export; pass --force to replace it"
    shutil.rmtree(out)
    return None


def export(repo_root: Path, out: Path, data_root: Path, author_name: str, author_email: str, private_root: Path | None = None,
           keep_config_paths: bool = False, gate_fn=None, dry_run: bool = False, force: bool = False) -> int:
    """Whole S11 flow; returns the exit code (0 ok, 2 refused / failed check)."""
    if not author_email.lower().endswith(NOREPLY):
        print(f"author e-mail must be a GitHub noreply address (ending in {NOREPLY})")
        return 2
    try:
        if dry_run:
            sel = select_files(repo_root)
            print(f"dry-run: would export {len(sel['files'])} files from {len(EXPORT_ITEMS)} items to {out} "
                  f"({sel['ignored']} excluded by .gitignore, {len(sel['missing_items'])} items missing)")
            return 0
        msg = prepare_out(out, force)
        if msg:
            print(mask(msg))
            return 2
        out.mkdir(parents=True)
        write_marker(out, status="started")
        counts = copy_tree(repo_root, out, keep_config_paths)
    except RuntimeError as exc:
        print(mask(str(exc)))
        return 2
    if counts["missing_items"]:
        print(f"missing repository items: {len(counts['missing_items'])} ({', '.join(counts['missing_items'])})")
        return 2
    sample = write_sample(out)
    print(f"copied {counts['files']} files ({counts['rewritten']} rewritten, {counts['replacements']} path replacements, "
          f"{counts['ignored']} excluded by .gitignore); sample: {sample['n_drives']} drives, {sample['n_windows']} windows, "
          f"{sample['bytes']} bytes from {sample['source']}")
    if sample["too_big"]:
        print(f"sample/ is {sample['bytes']} bytes (> {SAMPLE_MAX_BYTES}); nothing committed")
        write_marker(out, status="refused", reason="sample_too_big")
        return 2
    findings = precheck(out)
    if findings:
        print(f"pre-commit scan: {len(findings)} finding(s); nothing committed")
        for path, rule, n in findings[:60]:
            print(f"  {rule} {path} x{n}")
        write_marker(out, status="refused", reason="pre_commit_scan", findings=len(findings))
        return 2
    sha = git_init_commit(out, author_name, author_email, f"VTT build pipeline export ({C.CONFIG_VERSION}, schema {C.RELEASE_VERSION})")
    print(f"committed {sha[:12]} on main (author {mask(author_email)})")
    n_tracked, n_diff, n_ignored = tracked_mismatch(out)
    if n_diff or n_ignored:
        print(f"export / commit mismatch: {n_tracked} tracked files, {n_diff} differ from the export directory, "
              f"{n_ignored} on disk but excluded by .gitignore; fix .gitignore and re-run; do not push")
        write_marker(out, status="refused", reason="tracked_mismatch", commit=sha)
        return 2
    write_marker(out, status="committed", commit=sha, files=n_tracked, sample_bytes=sample["bytes"])
    ok, lines = run_gate(out, data_root, private_root, gate_fn)
    print("\n".join(lines))
    if not ok:
        print("code gate FAILED: fix the findings and re-run the export; do not push")
        write_marker(out, status="gate_failed", commit=sha, files=n_tracked)
        return 2
    write_marker(out, status="ok", commit=sha, files=n_tracked, sample_bytes=sample["bytes"])
    print(push_instructions(out, github_ssh_aliases()))
    return 0


def github_owner() -> str:
    """The account that owns config.GITHUB_REPO (default commit author name)."""
    return C.GITHUB_REPO.split("/")[0]


def default_author_email() -> str:
    """<owner>@users.noreply.github.com for the configured GitHub repository. GitHub links a commit to the account only
    for a noreply address enabled under Settings > Emails; the modern form is <id>+<user>@users.noreply.github.com
    (override with --author-email)."""
    return github_owner() + NOREPLY


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None, help="default <data-root>/github_export")
    ap.add_argument("--private-root", type=Path, default=None, help="default <data-root>/private when readable")
    ap.add_argument("--author-name", default=github_owner(),
                    help="commit author name (default: the GitHub account that owns config.GITHUB_REPO)")
    ap.add_argument("--author-email", default=default_author_email(),
                    help="commit author e-mail; must end in " + NOREPLY + ". GitHub attributes the commit only when the "
                         "address is the account's noreply address (Settings > Emails), today of the form "
                         "<id>+<user>" + NOREPLY + "; default <owner>" + NOREPLY)
    ap.add_argument("--keep-config-paths", action="store_true", help="leave vtt/config.py defaults untouched (gate will flag them)")
    ap.add_argument("--force", action="store_true", help="replace a non-empty --out that is not a previous export")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or (a.data_root / "github_export")
    private_root = a.private_root or (a.data_root / "private")
    if not (private_root.is_dir() and os.access(private_root, os.R_OK)):
        private_root = None
    return export(a.repo_root, out, a.data_root, a.author_name, a.author_email, private_root, a.keep_config_paths,
                  dry_run=a.dry_run, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
