"""S11 GitHub export: path rewriting, .gitignore-aware selection with git's own semantics, the export of the real
repository into a temporary data root through the real pre-commit scan and the real code gate, refusal on a planted
identifier, marker-based idempotence (a refused run is replaceable without --force), exit codes, kept file modes, the
tracked-files check, the author e-mail rule and the push instructions."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from vtt import config as C
from vtt.paths import P


def load_s11():
    path = Path(__file__).resolve().parents[1] / "scripts" / "s11_github_export.py"
    spec = importlib.util.spec_from_file_location("s11_github_export", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S11 = load_s11()
REPO_ROOT = Path(__file__).resolve().parents[1]


def noreply(user: str) -> str:
    return user + "@" + "users.noreply.github.com"


def hex16(n: int = 1) -> str:
    return "f" * 10 + "%06x" % n


@pytest.fixture()
def root():
    P.TMP.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="vtt_s11_", dir=P.TMP))
    yield d
    if not os.environ.get("VTT_KEEP_TMP"):
        shutil.rmtree(d, ignore_errors=True)


def lenient_gate(out: Path):
    return True, ["fake gate: ok"]


def make_fake_repo(root: Path, leak: str | None = None, gitignore: str = "__pycache__/\n", extra: dict[str, str] | None = None) -> Path:
    """The smallest repository with every EXPORT_ITEM present (and an importable vtt.config); `extra` = {rel: text}."""
    repo = root / "repo"
    for d in ("vtt", "scripts", "starter_kit", "experiments", "tests", "docs"):
        (repo / d).mkdir(parents=True)
    (repo / "vtt" / "__init__.py").write_text("")
    (repo / "vtt" / "config.py").write_text(f'DATA_ROOT = "{C.DATA_ROOT}"\nCONFIG_VERSION = "test"\n')
    (repo / "scripts" / "s00_demo.py").write_text("print('demo')\n")
    (repo / "starter_kit" / "__init__.py").write_text("")
    (repo / "experiments" / "__init__.py").write_text("")
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "docs" / "NOTES.md").write_text(f"Data live under `{C.DATA_ROOT}/build` and the raw library `{C.RAW_ROOT}`.\n")
    if leak:
        (repo / "docs" / "leak.md").write_text(f"device {leak} appears here\n")
    (repo / "README.md").write_text("# fake\n\nA driving automation system (DAS) data pipeline.\n")
    (repo / "LICENSE").write_text("MIT\n")
    (repo / "THIRD_PARTY_NOTICES.md").write_text("none\n")
    (repo / "requirements.txt").write_text("numpy\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    (repo / ".gitignore").write_text(gitignore)
    (repo / "vtt" / "__pycache__").mkdir()
    (repo / "vtt" / "__pycache__" / "config.cpython-312.pyc").write_bytes(b"\0\0")
    for rel, text in (extra or {}).items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    return repo


def git(out: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(out), *args], check=True, capture_output=True, text=True).stdout.strip()


def tracked(out: Path) -> set[str]:
    return {p for p in git(out, "ls-files", "-z").split("\0") if p}


def on_disk(out: Path) -> set[str]:
    return {p.relative_to(out).as_posix() for p in out.rglob("*")
            if p.is_file() and p.name != S11.MARKER and ".git" not in p.relative_to(out).parts}


def marker(out: Path) -> dict:
    return json.loads((out / S11.MARKER).read_text())


def test_rewrite_text_placeholders():
    text = f"root {C.DATA_ROOT}/build, raw {C.RAW_ROOT}/x, repo {REPO_ROOT}/vtt, home {Path.home()}/other, plan (internal design plan, not published) end"
    out, n = S11.rewrite_text(text)
    assert "$VTT_DATA_ROOT/build" in out and "$VTT_RAW_ROOT/x" in out and "$R/vtt" in out and "$HOME/other" in out
    assert "~/.claude" not in out and "(internal design plan, not published)" in out
    assert str(C.DATA_ROOT) not in out and str(Path.home()) not in out and n >= 5
    assert S11.rewrite_text("nothing to do") == ("nothing to do", 0)
    assert set(S11.PLACEHOLDERS) >= {"$R", "$VTT_DATA_ROOT", "$VTT_RAW_ROOT", "$PY", "$CPY", "$HOME"}
    readme = (REPO_ROOT / "README.md").read_text()
    for ph in S11.PLACEHOLDERS:                                   # every placeholder the rewrite can emit is documented
        assert f"`{ph}`" in readme, ph


def test_git_ignored_follows_git_semantics():
    gi = "*.csv\n!keep/*.csv\ndocs/secret.md\ndata/\n!sample/**\n"
    paths = ["vtt/a.csv", "keep/b.csv", "docs/secret.md", "docs/ok.md", "x/data/y.txt", "sample/data/v/d.parquet", ".gitignore"]
    ign = S11.git_ignored(gi, paths)
    assert ign == {"vtt/a.csv", "docs/secret.md", "x/data/y.txt"}          # a re-included path is never reported
    assert S11.git_ignored("", paths) == set() and S11.git_ignored(gi, []) == set()
    # the repository's own .gitignore: the synthetic sample is tracked, research sheets next to the spec tables are not
    real = (REPO_ROOT / ".gitignore").read_text()
    probe = ["sample/data/vehicle_0001/drive_000001.parquet", "sample/index/drives.parquet", "vtt/specs_data/platform_specs.csv",
             "vtt/specs_data/fingerprint_to_platform.csv", "vtt/specs_data/specs_research.csv", "vtt/specs_data/anything_else.csv",
             "build/x.txt", "vtt/private.py", "scripts/s11_github_export.py"]
    ign = S11.git_ignored(real, probe)
    assert ign == {"vtt/specs_data/specs_research.csv", "vtt/specs_data/anything_else.csv", "build/x.txt"}


def test_export_real_repo(root, capsys):
    """The real repository through the real pre-commit scan and the real code gate (no monkeypatching)."""
    pytest.importorskip("vtt.gate.pre_push")
    out = root / "github_export"
    rc = S11.export(REPO_ROOT, out, root, "tester", noreply("tester"))
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "ok=True" in text and "pre-commit scan" not in text
    for item in S11.EXPORT_ITEMS:
        assert (out / item).exists(), item
    assert not list(out.rglob("__pycache__")) and not list(out.rglob("*.pyc"))
    sample_bytes = sum(p.stat().st_size for p in (out / "sample").rglob("*") if p.is_file())
    assert 0 < sample_bytes <= S11.SAMPLE_MAX_BYTES
    assert (out / "sample" / "index" / "drives.parquet").exists() and len(list((out / "sample" / "data").rglob("*.parquet"))) == 4
    for p in (out / "docs").rglob("*.md"):
        t = p.read_text()
        assert str(C.DATA_ROOT) not in t and str(Path.home()) not in t, p.name
    cfg = (out / "vtt" / "config.py").read_text()
    assert str(C.DATA_ROOT) not in cfg and str(Path.home()) not in cfg and "$VTT_DATA_ROOT" in cfg
    env = dict(os.environ, PYTHONPATH=str(out), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([os.sys.executable, "-c", "import vtt.config as c; print(c.CONFIG_VERSION)"], cwd=str(out), env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == C.CONFIG_VERSION
    assert git(out, "rev-list", "--count", "HEAD") == "1" and git(out, "log", "-1", "--format=%ae") == noreply("tester")
    assert git(out, "branch", "--show-current") == "main"
    assert marker(out)["status"] == "ok" and marker(out)["commit"] == git(out, "rev-parse", "HEAD")
    # the commit holds exactly the export directory: sample data included, nothing git-ignored left on disk
    tr = on_disk(out)
    assert tracked(out) == tr and S11.tracked_mismatch(out) == (len(tr), 0, 0)
    assert sum(1 for p in tr if p.startswith("sample/data/") and p.endswith(".parquet")) == 4
    assert {"vtt/specs_data/platform_specs.csv", "vtt/specs_data/fingerprint_to_platform.csv"} <= tr
    assert not [p for p in tr if p.startswith("vtt/specs_data/") and ("research" in p or "fill" in p)]
    for p in (REPO_ROOT / "vtt" / "specs_data").glob("*.csv"):                     # research / fill sheets never leave the machine
        if p.name not in ("platform_specs.csv", "fingerprint_to_platform.csv"):
            assert not (out / "vtt" / "specs_data" / p.name).exists(), p.name
    for p in (REPO_ROOT / "scripts").iterdir():                                      # modes survive the text rewrite
        if p.is_file() and p.suffix in (".py", ".sh"):
            assert stat.S_IMODE((out / "scripts" / p.name).stat().st_mode) == stat.S_IMODE(p.stat().st_mode), p.name
    assert "git remote add origin 'git@<alias>:" + C.GITHUB_REPO + ".git'" in text and "ssh -T git@" + "github.com" in text
    assert "git push -u origin main" in text


def test_planted_token_refused_then_replaceable(root, capsys):
    repo = make_fake_repo(root, leak=hex16(3))
    out = root / "export"
    rc = S11.export(repo, out, root, "tester", noreply("tester"), gate_fn=lenient_gate)
    text = capsys.readouterr().out
    assert rc == 2 and "HEX16" in text and "nothing committed" in text and not (out / ".git").exists()
    assert hex16(3) not in text
    assert marker(out)["status"] == "refused" and marker(out)["findings"] == 1
    (repo / "docs" / "leak.md").unlink()                                             # fix and re-run: no --force needed
    assert S11.export(repo, out, root, "tester", noreply("tester"), gate_fn=lenient_gate) == 0
    assert marker(out)["status"] == "ok" and git(out, "rev-list", "--count", "HEAD") == "1"


def test_clean_fake_repo_exports_and_reexports(root, capsys):
    repo = make_fake_repo(root)
    out = root / "export"
    assert S11.export(repo, out, root, "tester", noreply("tester"), gate_fn=lenient_gate) == 0
    notes = (out / "docs" / "NOTES.md").read_text()
    assert "$VTT_DATA_ROOT/build" in notes and "$VTT_RAW_ROOT" in notes
    assert not (out / "vtt" / "__pycache__").exists()
    assert S11.export(repo, out, root, "tester", noreply("tester"), gate_fn=lenient_gate) == 0   # idempotent re-export
    assert git(out, "rev-list", "--count", "HEAD") == "1" and marker(out)["status"] == "ok"
    (root / "other").mkdir(); (root / "other" / "keep.txt").write_text("x")
    assert S11.export(repo, root / "other", root, "tester", noreply("tester"), gate_fn=lenient_gate) == 2   # refused, exit 2
    assert "pass --force" in capsys.readouterr().out and (root / "other" / "keep.txt").exists()
    assert S11.export(repo, root / "other", root, "tester", noreply("tester"), gate_fn=lenient_gate, force=True) == 0


def test_gitignore_excludes_and_modes_kept(root, capsys):
    gi = "__pycache__/\n*.csv\n!docs/keep.csv\ndocs/secret.md\ndata/\n!sample/**\n"      # data/ must not drop the sample
    extra = {"vtt/table.csv": "a,b\n1,2\n", "docs/keep.csv": "x\n1\n", "docs/secret.md": f"device {hex16(5)}\n",
             "vtt/data/cache.txt": "x\n", "scripts/run.sh": "#!/bin/sh\necho hi\n"}
    repo = make_fake_repo(root, gitignore=gi, extra=extra)
    (repo / "scripts" / "run.sh").chmod(0o755)
    out = root / "export"
    assert S11.export(repo, out, root, "tester", noreply("tester"), gate_fn=lenient_gate) == 0
    text = capsys.readouterr().out
    assert "3 excluded by .gitignore" in text
    for rel in ("vtt/table.csv", "docs/secret.md", "vtt/data/cache.txt"):
        assert not (out / rel).exists(), rel
    assert (out / "docs" / "keep.csv").exists() and (out / "scripts" / "run.sh").exists()
    assert stat.S_IMODE((out / "scripts" / "run.sh").stat().st_mode) & stat.S_IXUSR
    assert tracked(out) == on_disk(out) and "docs/keep.csv" in tracked(out)
    assert S11.export(repo, root / "dry", root, "tester", noreply("tester"), dry_run=True) == 0
    assert "3 excluded by .gitignore" in capsys.readouterr().out and not (root / "dry").exists()


def test_tracked_mismatch_refused(root, capsys):
    repo = make_fake_repo(root, gitignore="__pycache__/\nsample/\n")                 # the old bug: the sample would be dropped
    out = root / "export"
    rc = S11.export(repo, out, root, "tester", noreply("tester"), gate_fn=lenient_gate)
    text = capsys.readouterr().out
    assert rc == 2 and "mismatch" in text and "excluded by .gitignore" in text and "git push" not in text
    assert marker(out)["status"] == "refused" and marker(out)["reason"] == "tracked_mismatch"


def test_sample_too_big_refused(root, monkeypatch, capsys):
    repo = make_fake_repo(root)
    monkeypatch.setattr(S11, "SAMPLE_MAX_BYTES", 10)
    rc = S11.export(repo, root / "export", root, "tester", noreply("tester"), gate_fn=lenient_gate)
    assert rc == 2 and "nothing committed" in capsys.readouterr().out and not (root / "export" / ".git").exists()
    assert marker(root / "export")["reason"] == "sample_too_big"


def test_author_defaults_email_rule_and_dry_run(root, capsys):
    repo = make_fake_repo(root)
    assert S11.export(repo, root / "e1", root, "tester", "tester@" + "example.org", gate_fn=lenient_gate) == 2
    assert not (root / "e1").exists()
    assert S11.export(repo, root / "e2", root, "tester", noreply("tester"), dry_run=True) == 0
    assert not (root / "e2").exists() and "dry-run" in capsys.readouterr().out
    assert S11.default_author_email() == S11.github_owner() + S11.NOREPLY and S11.github_owner() == C.GITHUB_REPO.split("/")[0]
    help_text = subprocess.run([os.sys.executable, str(REPO_ROOT / "scripts" / "s11_github_export.py"), "--help"],
                               capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT))).stdout
    assert "<id>+<user>" + S11.NOREPLY in help_text


def test_gate_failure_blocks_push_instructions(root, capsys):
    repo = make_fake_repo(root)
    rc = S11.export(repo, root / "export", root, "tester", noreply("tester"), gate_fn=lambda o: (False, ["gate: findings=1"]))
    text = capsys.readouterr().out
    assert rc == 2 and "code gate FAILED" in text and "git push" not in text
    assert marker(root / "export")["status"] == "gate_failed"


def test_real_gate_on_clean_fake_repo(root, capsys):
    pytest.importorskip("vtt.gate.pre_push")
    repo = make_fake_repo(root)
    rc = S11.export(repo, root / "export", root, "tester", noreply("tester"))
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "ok=True" in text and "git push -u origin main" in text


def test_missing_gate_module_fails_loudly(root, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_gate(name, *a, **k):
        if name.startswith("vtt.gate"):
            raise ImportError("gate absent")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_gate)
    ok, lines = S11.run_gate(root, root, None)
    assert ok is False and any("GATE MODULE MISSING" in ln for ln in lines)


def test_push_instructions_and_aliases():
    text = S11.push_instructions(Path("x"), ["github.com", "github-two"])
    assert "ssh -T git@github-two" in text and "'git@<alias>:" + C.GITHUB_REPO + ".git'" in text
    assert "github.com" in S11.github_ssh_aliases()
