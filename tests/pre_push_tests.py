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


def test_figure_source_csv_exempt_only_there(root):
    repo = make_repo(root)
    commit_file(repo, "paper/figure_source/source_data/panel_a/points.csv", "x,y\n1,2\n", "figure data")
    commit_file(repo, "paper/figure_source/source_data/cache.parquet", b"PAR1", "parquet next to the figure data")
    commit_file(repo, "paper/other/table.csv", "x,y\n1,2\n", "csv elsewhere")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    typed = {f.path for f in rep.findings if f.rule in ("FILE_TYPE", "HISTORY_TYPE")}
    assert "paper/figure_source/source_data/panel_a/points.csv" not in typed
    assert {"paper/figure_source/source_data/cache.parquet", "paper/other/table.csv"} <= typed, rep.counts()


def test_figure_source_csv_text_still_scanned(root):
    repo = make_repo(root)
    commit_file(repo, "paper/figure_source/source_data/leak.csv", f"device\n{hex16()}\n", "leaky figure data")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert any(f.rule == "HEX16" and f.path.endswith("leak.csv") for f in rep.findings), rep.counts()


def _pdf_with_text(path: Path, text: str, author: str = "") -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3, 1))
    fig.text(0.05, 0.5, text, fontsize=8)
    fig.savefig(path, metadata={"Creator": "test", "Producer": "test", "Author": author, "CreationDate": None})
    plt.close(fig)
    return path.read_bytes()


def _mini_pdf(text: str, xmp: str | None = None, info: dict | None = None, noise: bytes | None = None,
              objstm: bytes | None = None, objstm_filter: str = "/FlateDecode", trailer_extra: str = "") -> bytes:
    """A small valid PDF built by hand: one page with `text`, optional XMP packet, Info dict, an unreferenced stream,
    an unreferenced object stream (Flate-compressed `objstm` body) and extra trailer keys."""
    import zlib
    objs = []
    cat = "<< /Type /Catalog /Pages 2 0 R" + (" /Metadata 5 0 R" if xmp is not None else "") + " >>"
    objs.append(cat.encode())
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Contents 4 0 R "
                b"/Resources << /Font << /F1 6 0 R >> >> >>")
    content = f"BT /F1 10 Tf 10 50 Td ({text}) Tj ET".encode()
    objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    x = (xmp or "").encode()
    objs.append(b"<< /Type /Metadata /Subtype /XML /Length %d >>\nstream\n" % len(x) + x + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    n = noise or b""
    objs.append(b"<< /Length %d >>\nstream\n" % len(n) + n + b"\nendstream")
    objs.append(("<< " + " ".join(f"/{k} ({v})" for k, v in (info or {}).items()) + " >>").encode())
    if objstm is not None:
        body = zlib.compress(objstm) if objstm_filter == "/FlateDecode" else objstm
        objs.insert(len(objs) - 1, b"<< /Type /ObjStm /N 1 /First 4 /Filter %s /Length %d >>\nstream\n"
                    % (objstm_filter.encode(), len(body)) + body + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for k, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % k + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R %s>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, len(objs), trailer_extra.encode(), xref)
    return bytes(out)


NO_ALLOW = {"email": "", "iso_date_files": [], "pre_push_iso_date_files": []}
needs_poppler = pytest.mark.skipif(not (shutil.which("pdftotext") and shutil.which("pdfinfo")), reason="poppler tools missing")


@needs_poppler
def test_pdf_text_layer_is_scanned(root):
    repo = make_repo(root)
    address = "someone" + "@" + "example.org"
    data = _pdf_with_text(root / "a.pdf", f"write to {address}")
    commit_file(repo, "paper/figures/a.pdf", data, "pdf with an address in its text")
    assert address in (pre_push.scan_text("paper/figures/a.pdf", data) or "")
    rep = pre_push.run(repo, private_root=None, allow=NO_ALLOW, python=sys.executable)
    assert any(f.rule == "EMAIL" and f.path == "paper/figures/a.pdf" for f in rep.findings), rep.counts()


@needs_poppler
def test_pdf_info_metadata_is_scanned(root):
    repo = make_repo(root)
    data = _pdf_with_text(root / "m.pdf", "plain figure", author=f"device {hex16(7)}")
    commit_file(repo, "paper/figures/m.pdf", data, "pdf with an identifier in its metadata")
    rep = pre_push.run(repo, private_root=None, allow=NO_ALLOW, python=sys.executable)
    assert any(f.rule == "HEX16" and f.path == "paper/figures/m.pdf" for f in rep.findings), rep.counts()


@needs_poppler
def test_pdf_xmp_custom_keys_and_trailing_bytes_are_scanned(root):
    xmp = f'<x:xmpmeta><rdf:Description xmpMM:DocumentID="{hex16(11)}"/></x:xmpmeta>'
    path = "/" + "home" + "/someone/figure.svg"
    data = _mini_pdf("plain figure", xmp=xmp, info={"SourceFile": path})
    t = pre_push.scan_text("paper/figures/x.pdf", data)
    assert hex16(11) in t and path in t and "plain figure" in t
    tail = data + b"\n% " + hex16(12).encode() + b"\n"
    assert hex16(12) in pre_push.scan_text("paper/figures/x.pdf", tail)
    repo = make_repo(root)
    commit_file(repo, "paper/figures/x.pdf", data, "pdf with identifiers outside its page text")
    rules = {f.rule for f in pre_push.run(repo, private_root=None, allow=NO_ALLOW, python=sys.executable).findings}
    assert {"HEX16", "PATHS"} <= rules, rules


@needs_poppler
def test_pdf_stream_noise_is_not_text(root):
    noise = b"x" + b"@" + b"y1.db " + ("A1" * 9).encode()                  # e-mail- / VIN-shaped bytes inside a stream
    data = _mini_pdf("plain figure", noise=noise)
    t = pre_push.scan_text("paper/figures/n.pdf", data)
    assert t is not None and "plain figure" in t and "y1.db" not in t
    assert pre_push.pdf_has_page_text(t)


def test_unreadable_pdf_is_a_finding(root):
    repo = make_repo(root)
    commit_file(repo, "paper/figures/broken.pdf", b"%PDF-1.4 not really a pdf\n", "broken pdf")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert any(f.rule == "PDF_UNREADABLE" for f in rep.findings), rep.counts()
    assert any(f.rule == "HISTORY_PDF_UNREADABLE" for f in rep.findings), rep.counts()


def test_pdf_detected_by_content_not_name(root):
    repo = make_repo(root)
    commit_file(repo, "docs/figure.bin", b"%PDF-1.4 not really a pdf\n", "pdf bytes under another name")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert any(f.rule == "PDF_UNREADABLE" for f in rep.findings), rep.counts()


def test_figure_source_csv_must_be_utf8(root):
    repo = make_repo(root)
    commit_file(repo, "paper/figure_source/source_data/u16.csv", f"device\n{hex16(3)}\n".encode("utf-16"), "utf-16 csv")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert any(f.rule == "UNREADABLE" and f.path.endswith("u16.csv") for f in rep.findings), rep.counts()


def test_suffix_rules_ignore_case_and_geodata_is_pinned(root):
    repo = make_repo(root)
    commit_file(repo, "data2/table.CSV", "a,b\n1,2\n", "upper-case csv")
    commit_file(repo, "maps/trace.geojson", '{"type": "FeatureCollection", "features": []}', "geojson")
    pinned = next(iter(pre_push.PINNED_FILES))
    commit_file(repo, pinned, '{"type": "FeatureCollection", "features": []}', "not the pinned outline")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    typed = {(f.rule, f.path) for f in rep.findings}
    assert ("FILE_TYPE", "data2/table.CSV") in typed and ("FILE_TYPE", "maps/trace.geojson") in typed, typed
    assert any(r == "FILE_TYPE" and p == pinned for r, p in typed), typed



@needs_poppler
def test_pdf_object_streams_are_inflated_and_scanned(root):
    link = b"10 0 << /Type /Annot /Subtype /Link /A << /URI (file:///" + b"home" + b"/someone/" + hex16(21).encode() + b") >> >>"
    data = _mini_pdf("plain figure", objstm=link)
    t = pre_push.scan_text("paper/compiled/x.pdf", data)
    assert hex16(21) in t and "plain figure" in t
    repo = make_repo(root)
    commit_file(repo, "paper/compiled/x.pdf", data, "pdf with an identifier in a compressed object stream")
    rules = {f.rule for f in pre_push.run(repo, private_root=None, allow=NO_ALLOW, python=sys.executable).findings}
    assert {"HEX16", "PATHS"} <= rules, rules


@needs_poppler
def test_pdf_object_stream_with_unknown_filter_fails_closed(root):
    data = _mini_pdf("plain figure", objstm=b"10 0 (x)", objstm_filter="/LZWDecode")
    assert pre_push.scan_text("paper/figures/l.pdf", data) == pre_push.UNREADABLE_PDF


def test_encrypted_pdf_is_a_finding(root):
    repo = make_repo(root)
    commit_file(repo, "paper/figures/e.pdf", _mini_pdf("plain figure", trailer_extra="/Encrypt 99 0 R "), "encrypted pdf")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert any(f.rule == "PDF_ENCRYPTED" for f in rep.findings), rep.counts()


def test_text_mentioning_the_pdf_magic_is_not_a_pdf(root):
    repo = make_repo(root)
    commit_file(repo, "docs/formats.md", "# Formats\n\nA PDF file starts with the bytes %PDF- and a version.\n", "doc")
    rep = pre_push.run(repo, private_root=None, python=sys.executable)
    assert not any(f.rule.endswith("PDF_UNREADABLE") for f in rep.findings), rep.counts()
