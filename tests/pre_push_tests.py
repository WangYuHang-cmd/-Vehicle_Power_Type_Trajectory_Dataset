"""Pre-push gate on temporary git repositories: planted tokens in history, forbidden types, data directories,
the tests/ no-allow-list rule, size limits, config exposure and author e-mails."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from vtt.gate import pre_push
from vtt.paths import P


def hex16(n: int = 1) -> str:
    return "f" * 10 + "%06x" % n


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_repo(root: Path, config_text: str = "CONFIG_VERSION = 'test'\n") -> Path:
    repo = root / "repo"
    (repo / "vtt").mkdir(parents=True)
    (repo / "vtt" / "__init__.py").write_text("")
    (repo / "vtt" / "config.py").write_text(config_text)
    (repo / "README.md").write_text("# test repository\n\nNothing sensitive here.\n")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "tester@" + "users.noreply.github.com")
    git(repo, "config", "user.name", "tester")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture()
def root():
    P.TMP.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="vtt_push_", dir=P.TMP))
    yield d
    if not os.environ.get("VTT_KEEP_TMP"):
        shutil.rmtree(d, ignore_errors=True)


def commit_file(repo: Path, rel: str, content, msg: str = "add") -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)


def test_clean_repo_passes(root):
    repo = make_repo(root)
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert rep.ok, rep.counts()
    assert rep.n_files == 3


def test_planted_token_in_history_fails(root):
    repo = make_repo(root)
    commit_file(repo, "notes.txt", f"device {hex16()}\n", "leak")
    git(repo, "rm", "-q", "notes.txt")
    git(repo, "commit", "-q", "-m", "remove")
    assert not (repo / "notes.txt").exists()
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert not rep.ok
    assert any(f.rule == "HISTORY_HEX16" for f in rep.findings), rep.counts()
    assert hex16() not in str(rep.to_dict())


def test_forbidden_types_and_data_dirs(root):
    repo = make_repo(root)
    commit_file(repo, "cache.parquet", b"PAR1", "parquet")
    commit_file(repo, "vtt/specs_data/platform_specs.csv", "platform,brand\nX,Y\n", "specs")
    commit_file(repo, "tests/fixtures/a.csv", "a,b\n1,2\n", "fixture")
    commit_file(repo, "release/v1/README.md", "# release\n", "release dir")
    commit_file(repo, "vtt/private.py", "x = 1\n", "module named private")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    rules = {(f.rule, f.path) for f in rep.findings}
    assert ("FILE_TYPE", "cache.parquet") in rules and ("HISTORY_TYPE", "cache.parquet") in rules
    assert ("DATA_DIR", "release/v1/README.md") in rules
    assert not any(p.startswith(("vtt/specs_data", "tests/fixtures", "vtt/private.py")) for _, p in rules), rules


def test_tests_dir_has_no_allowlist(root):
    repo = make_repo(root)
    date = f"{2024}-{5:02d}-{6:02d}"
    commit_file(repo, "docs/a.md", f"written {date}\n", "doc")
    commit_file(repo, "tests/t_tests.py", f"DATE = '{date}'\n", "test")
    allow = {"email": "", "iso_date_files": [], "pre_push_iso_date_files": ["docs/a.md", "tests/t_tests.py"]}
    rep = pre_push.run(repo, private_root=None, allow=allow, python=sys.executable)
    paths = {f.path for f in rep.findings if f.rule in ("ISO_DATE", "HISTORY_ISO_DATE")}
    assert paths == {"tests/t_tests.py"}, rep.counts()


def test_config_exposure_is_a_finding(root):
    repo = make_repo(root, "CONFIG_VERSION = 'test'\nLAB_DONGLES = ()\n")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    rules = {f.rule for f in rep.findings}
    assert {"CONFIG_TEXT", "CONFIG_EXPOSES"} <= rules, rep.counts()


def test_large_file_is_a_finding(root):
    repo = make_repo(root)
    commit_file(repo, "big.bin", b"\0" * (2 * 1024 * 1024 + 1), "big")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    rules = {f.rule for f in rep.findings}
    assert {"FILE_SIZE", "BLOB_SIZE"} <= rules, rep.counts()


def test_author_email_against_forbidden_tokens(root):
    repo = make_repo(root)
    priv = root / "private"
    priv.mkdir()
    (priv / "forbidden_tokens.txt").write_text("tester\n")
    rep = pre_push.run(repo, private_root=priv, python=sys.executable)
    assert any(f.rule == "AUTHOR_EMAIL" for f in rep.findings), rep.counts()
    assert "tester" not in " ".join(f.detail for f in rep.findings if f.rule == "AUTHOR_EMAIL")


def test_not_a_repo(root):
    rep = pre_push.run(root, private_root=None)
    assert [f.rule for f in rep.findings] == ["NOT_A_REPO"]


def test_coord_rule_applies_to_names_not_to_source_prose(root):
    """Source and docs may name the private coordinate / origin columns; file names, CSV headers, JSON keys and the
    column names of tracked parquet files may not. Identifier rules stay on all text."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    repo = make_repo(root)
    words = ", ".join(("coun" + "try", "in_" + "house", "la" + "t", "lo" + "n", "g" + "ps"))
    commit_file(repo, "vtt/labels.py", f'PRIVATE_COLS = ("{words}")  # columns of the private route master\n', "code")
    commit_file(repo, "docs/NOTES.md", f"The private table carries {words}; none of them is released.\n", "docs")
    commit_file(repo, "tests/labels_tests.py", f"WORDS = '{words}'\n", "tests")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert rep.ok, rep.counts()
    commit_file(repo, "docs/" + "g" + "ps_notes.md", "nothing\n", "named file")
    commit_file(repo, "vtt/specs_data/specs.csv", "platform,coun" + "try\nX,Y\n", "csv header")
    commit_file(repo, "vtt/gate/allow.json", '{"la' + 't": 1}\n', "json key")
    table = pa.table({"t_s": pa.array([0.0, 0.1]), "logMono" + "Time": pa.array([1, 2], pa.int64())})
    (repo / "sample").mkdir()
    pq.write_table(table, repo / "sample" / "drive.parquet")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "sample")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    got = {(f.rule, f.path, f.column) for f in rep.findings}
    assert ("COORD", "docs/" + "g" + "ps_notes.md", "<path>") in got, rep.counts()
    assert ("HISTORY_COORD", "docs/" + "g" + "ps_notes.md", "<path>") in got
    assert ("COORD", "vtt/specs_data/specs.csv", "<columns>") in got and ("HISTORY_COORD", "vtt/specs_data/specs.csv", "<columns>") in got
    assert ("COORD", "vtt/gate/allow.json", "<json_keys>") in got
    assert ("MONO", "sample/drive.parquet", "<columns>") in got
    assert not any(p in ("vtt/labels.py", "docs/NOTES.md", "tests/labels_tests.py") for _, p, _ in got), got
    commit_file(repo, "vtt/labels.py", f"X = '{hex16(2)}'\n", "leak")           # identifier rules unchanged
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert ("HEX16", "vtt/labels.py", "") in {(f.rule, f.path, f.column) for f in rep.findings}


def blob_sha(repo: Path, rev_path: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", rev_path], check=True, capture_output=True).stdout.decode().strip()


def test_accepted_history_finding_is_bound_to_the_blob(root):
    """A reviewed finding in already-published history can be accepted for one blob sha and one rule, with a reason;
    the gate then warns instead of failing. Another blob, a missing reason, another rule or a tests/ path do not."""
    repo = make_repo(root)
    commit_file(repo, "notes.md", "contact someone@" + "example.org for details\n", "leak")
    sha = blob_sha(repo, "HEAD:notes.md")
    commit_file(repo, "notes.md", "contact address removed\n", "fix")
    base = {"email": "", "iso_date_files": [], "pre_push_iso_date_files": []}
    rep = pre_push.run(repo, private_root=None, allow=dict(base), python=sys.executable)
    assert any(f.rule == "HISTORY_EMAIL" for f in rep.findings), rep.counts()
    dashed = "-".join(sha[i:i + 8] for i in range(0, 40, 8))          # the form used in gate_allow.json
    good = dict(base, history_accepted=[{"blob": dashed, "rule": "EMAIL", "reason": "example address in a test"}])
    rep = pre_push.run(repo, private_root=None, allow=good, python=sys.executable)
    assert not any(f.rule == "HISTORY_EMAIL" for f in rep.findings), rep.counts()
    assert any("accepted history finding EMAIL" in w for w in rep.warnings), rep.warnings
    rep = pre_push.run(repo, private_root=None, allow=dict(base, history_accepted=[{"blob": sha, "rule": "EMAIL", "reason": "plain sha"}]), python=sys.executable)
    assert not any(f.rule == "HISTORY_EMAIL" for f in rep.findings), rep.counts()
    for bad in ({"blob": "0" * 40, "rule": "EMAIL", "reason": "x"}, {"blob": sha, "rule": "EMAIL", "reason": ""},
                {"blob": sha, "rule": "PATHS", "reason": "x"}, {"blob": sha[:12], "rule": "EMAIL", "reason": "x"}):
        rep = pre_push.run(repo, private_root=None, allow=dict(base, history_accepted=[bad]), python=sys.executable)
        assert any(f.rule == "HISTORY_EMAIL" for f in rep.findings), bad
    # the same content under tests/ is never accepted
    commit_file(repo, "tests/notes_tests.py", "contact someone@" + "example.org for details\n", "tests copy")
    rep = pre_push.run(repo, private_root=None, allow=good, python=sys.executable)
    assert any(f.rule in ("EMAIL", "HISTORY_EMAIL") for f in rep.findings), rep.counts()
