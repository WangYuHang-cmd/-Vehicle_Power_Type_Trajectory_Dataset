"""Release-tree gate on a synthetic release: a clean package yields zero findings, a poisoned copy exactly four,
plus CHECKSUMS, metadata, enum / id-format, private-token and report-location behaviour."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.synth_build import make_build
from vtt import schema as S
from vtt.gate import pre_upload
from vtt.gate import rules as R
from vtt.gate import scan as SC
from vtt.package import Roots, package, write_checksums
from vtt.paths import P

META = {b"vtt_table": b"drive", b"vtt_schema_version": S.SCHEMA_VERSION.encode()}


def hex16(n: int = 1) -> str:
    return "f" * 10 + "%06x" % n


def hex12(n: int = 1) -> str:
    return "e" * 6 + "%06x" % n


@pytest.fixture(scope="module")
def site():
    P.TMP.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_scan_", dir=P.TMP))
    make_build(root / "build")
    roots = Roots.from_data_root(root)
    summary = package(roots, per_shard=4, workers=1, run_gate=False)
    assert summary["accepted"]
    yield root, roots
    if not os.environ.get("VTT_KEEP_TMP"):
        shutil.rmtree(root, ignore_errors=True)


def copy_release(site, name: str) -> Roots:
    root, roots = site
    dst = root / name / "release"
    shutil.copytree(roots.release, dst, ignore=shutil.ignore_patterns(".cache", ".gate_ok"))
    return Roots(root / name, roots.build, dst, root / name / "private")


def rewrite(path: Path, table: pa.Table) -> None:
    pq.write_table(table.replace_schema_metadata(META), path, compression="zstd")


def test_clean_release_has_zero_findings(site):
    _, roots = site
    rep = SC.scan_release(roots.release, private_root=None, workers=1)
    assert rep.findings == [], rep.counts()
    assert rep.n_files > 20 and any("private root" in w for w in rep.warnings)


def test_poisoned_release_yields_exactly_four_findings(site):
    r2 = copy_release(site, "poison")
    rel = r2.release
    with open(rel / "README.md", "a") as fh:
        fh.write(f"\nnote {hex16()}\n")                                      # 1: a 16-hex id in the card
    files = sorted((rel / "data").rglob("*.parquet"))
    t_a = pq.read_table(files[0])
    rewrite(files[0], t_a.append_column("extra", pa.array(np.zeros(t_a.num_rows, np.float32))))   # 2: extra column
    t_b = pq.read_table(files[1])
    clock = pa.array(10 ** 12 + np.arange(t_b.num_rows, dtype=np.int64) * 10 ** 8, pa.int64())
    rewrite(files[1], t_b.set_column(t_b.schema.get_field_index("seg"), "seg", clock))             # 3: int64 boot clock
    (rel / "notes.txt").write_text("release notes\n")                                                # 4: outside the allow-list
    write_checksums(r2, workers=1, use_cache=False)
    rep = SC.scan_release(rel, private_root=None, workers=1)
    kinds = sorted((f.gate, f.rule) for f in rep.findings)
    assert kinds == [("G0", "ALLOWLIST"), ("G1", "HEX16"), ("G3", "COLUMNS"), ("G5", "MAGNITUDE")], rep.counts()
    assert len(rep.findings) == 4
    dumped = json.dumps(rep.to_dict())
    assert R.HEX16.search(dumped) is None                                     # the report never carries the token
    assert any(f.path == "README.md" and f.count == 1 for f in rep.findings)
    assert any(f.rule == "MAGNITUDE" and f.column == "seg" for f in rep.findings)


def test_poisoned_release_with_workers_matches_sequential(site):
    r2 = copy_release(site, "poison_mp")
    (r2.release / "notes.txt").write_text("x\n")
    write_checksums(r2, workers=1, use_cache=False)
    seq = SC.scan_release(r2.release, private_root=None, workers=1)
    par = SC.scan_release(r2.release, private_root=None, workers=3)
    assert [f.to_dict() for f in seq.findings] == [f.to_dict() for f in par.findings]
    assert len(seq.findings) == 1 and seq.findings[0].rule == "ALLOWLIST"


def test_checksums_mismatch_missing_and_stale(site):
    r2 = copy_release(site, "cksum")
    rel = r2.release
    with open(rel / "LICENSE", "a") as fh:
        fh.write("\n")
    rep = SC.scan_release(rel, private_root=None, workers=1)
    assert [(f.gate, f.rule) for f in rep.findings] == [("CHECKSUMS", "MISMATCH")]
    lines = (rel / "CHECKSUMS.sha256").read_text().splitlines()
    kept = [ln for ln in lines if not ln.endswith("  LICENSE")] + ["0" * 64 + "  ghost/file.parquet"]
    (rel / "CHECKSUMS.sha256").write_text("\n".join(kept) + "\n")
    rep = SC.scan_release(rel, private_root=None, workers=1)
    assert sorted(f.rule for f in rep.findings) == ["NOT_LISTED", "STALE_ENTRY"]


def test_parquet_metadata_keys_enforced(site):
    r2 = copy_release(site, "meta")
    p = r2.release / "splits" / "train_uniform_size.parquet"
    pd.read_parquet(p).to_parquet(p, index=False)                            # pandas metadata sneaks in
    write_checksums(r2, workers=1, use_cache=False)
    rep = SC.scan_release(r2.release, private_root=None, workers=1)
    assert [(f.gate, f.rule) for f in rep.findings] == [("G4", "META_KEYS")]


def test_enum_domain_id_format_and_t_s(site):
    r2 = copy_release(site, "enum")
    rel = r2.release
    df = pd.read_parquet(rel / "index" / "drives.parquet")
    df.loc[df.index[0], "size_class"] = "HUGE"
    df.loc[df.index[1], "vehicle_id"] = "veh_1"
    S.write_public_parquet(df, rel / "index" / "drives.parquet", S.DRIVES_INDEX_COLUMNS, "drives")
    f0 = sorted((rel / "data").rglob("*.parquet"))[0]
    t = pq.read_table(f0)
    shifted = pa.array(np.asarray(t["t_s"].to_numpy()) + 1.0, pa.float64())
    rewrite(f0, t.set_column(t.schema.get_field_index("t_s"), "t_s", shifted))
    rep = SC.scan_release(rel, private_root=None, workers=1, checksums=False)
    kinds = sorted((f.gate, f.rule) for f in rep.findings)
    assert ("G6", "ENUM") in kinds and ("G8", "ID_FORMAT") in kinds and ("G5", "T_S") in kinds
    assert all(k[0] in ("G5", "G6", "G8") for k in kinds), kinds


def test_private_tokens_and_forbidden_words_are_findings(site):
    r2 = copy_release(site, "g9")
    priv = r2.private
    priv.mkdir(parents=True, exist_ok=True)
    (priv / "forbidden_tokens.txt").write_text("Zorblax\n")
    pd.DataFrame({"route_uid": [hex12(7)], "car_dir": ["CAR_X"], "dongle_dir": [hex16(7)], "dongle_log": [hex16(7)],
                  "route_id": ["a" * 8 + "--" + "b" * 10], "identity_uid": [hex12(6)], "vin_hash": [None]}).to_parquet(priv / "uid_map.parquet", index=False)
    with open(r2.release / "README.md", "a") as fh:
        fh.write(f"\nthanks to Zorblax for {hex12(7)}\n")
    write_checksums(r2, workers=1, use_cache=False)
    rep = SC.scan_release(r2.release, private_root=priv, workers=1)
    rules = sorted(f.rule for f in rep.findings)
    assert rules == ["EXACT", "FORBIDDEN", "HEX12"], rep.counts()
    assert not any("private root" in w for w in rep.warnings)
    clean = SC.scan_release(r2.release, private_root=priv / "nope", workers=1)
    assert sorted(f.rule for f in clean.findings) == ["HEX12"] and any("private root" in w for w in clean.warnings)


def test_missing_required_and_index_consistency(site):
    r2 = copy_release(site, "missing")
    rel = r2.release
    (rel / "LICENSE").unlink()
    next((rel / "data").rglob("*.parquet")).unlink()
    rep = SC.scan_release(rel, private_root=None, workers=1)
    kinds = {(f.gate, f.rule) for f in rep.findings}
    assert ("G0", "MISSING") in kinds and ("G0", "INDEX_FILES") in kinds and ("CHECKSUMS", "STALE_ENTRY") in kinds


def test_report_location_and_marker(site):
    root, roots = site
    rep = SC.scan_release(roots.release, private_root=None, workers=1)
    path = SC.write_report(rep, data_root=root, private_root=root / "no_private")
    assert path.parent == root / "tmp" and path.name.startswith("gate_") and json.loads(path.read_text())["ok"] is True
    priv = root / "private"
    priv.mkdir(exist_ok=True)
    path2 = SC.write_report(rep, data_root=root, private_root=priv)
    assert path2.parent == priv / "logs" and (path2.stat().st_mode & 0o777) == 0o600
    rep2, _ = pre_upload.run(roots.release, data_root=root, private_root=priv, workers=1)
    assert rep2.ok and json.loads((roots.release / ".gate_ok").read_text())["n_files"] == rep2.n_files
    r2 = copy_release(site, "marker")
    (r2.release / ".gate_ok").write_text("{}")
    (r2.release / "stray.bin").write_bytes(b"\0\1")
    rep3, _ = pre_upload.run(r2.release, data_root=r2.data, private_root=None, workers=1)
    assert not rep3.ok and not (r2.release / ".gate_ok").exists()


def test_allowlist_patterns():
    for ok in ("README.md", "LICENSE", "CHECKSUMS.sha256", ".gitattributes", ".gate_ok", "index/drives.parquet", "index/split_summary.csv",
               "index/readme_numbers.json", "splits/splits.json", "splits/eval_balanced_size_test.parquet", "schema/drive.json",
               "data/vehicle_0001/drive_000001.parquet", "windows/train/train-00000-of-00003.parquet"):
        assert SC.is_allowed(ok), ok
    for bad in ("notes.txt", "data/vehicle_1/drive_000001.parquet", "windows/dev/x.parquet", "index/sub/x.parquet", "work/x.parquet",
                "data/vehicle_0001/" + hex12() + ".parquet"):
        assert not SC.is_allowed(bad), bad


def test_index_and_splits_tables_are_scanned_in_full(site):
    """A planted token in the LAST row of an index table and of a splits table is found even when it lies beyond the
    string-value sample (which applies to drive tables and window shards only)."""
    r2 = copy_release(site, "fullscan")
    rel = r2.release
    df = pd.read_parquet(rel / "index" / "drives.parquet")
    assert len(df) == 8
    df.loc[df.index[-1], "label_note"] = f"see {hex16(5)}"
    S.write_public_parquet(df, rel / "index" / "drives.parquet", S.DRIVES_INDEX_COLUMNS, "drives")
    sp = rel / "splits" / "train_uniform_size.parquet"
    t = pq.read_table(sp)
    ids = t["window_id"].to_pylist()
    ids[-1] = hex16(6)
    t2 = t.set_column(0, "window_id", pa.array(ids, pa.string())).replace_schema_metadata(dict(pq.read_schema(sp).metadata))
    pq.write_table(t2, sp, compression="zstd")
    write_checksums(r2, workers=1, use_cache=False)
    rep = SC.scan_release(rel, private_root=None, workers=1, sample_rows=4)
    hits = {(f.path, f.gate, f.rule) for f in rep.findings}
    assert ("index/drives.parquet", "G2", "HEX16") in hits, rep.counts()
    assert ("splits/train_uniform_size.parquet", "G2", "HEX16") in hits and ("splits/train_uniform_size.parquet", "G8", "ID_FORMAT") in hits
    assert hex16(5) not in json.dumps(rep.to_dict()) and hex16(6) not in json.dumps(rep.to_dict())
    assert all(f.path in ("index/drives.parquet", "splits/train_uniform_size.parquet") for f in rep.findings), rep.counts()


def test_batched_scan_matches_whole_table_scan(site, monkeypatch):
    """With tiny batches (many row groups per file) the clean release is still clean and the poisoned copy still
    yields its four findings: the cross-batch t_s / magnitude / string logic does not depend on the batch size."""
    monkeypatch.setattr(SC, "SCAN_BATCH_BYTES", 1)                             # -> 256 rows per batch
    _, roots = site
    assert SC.batch_rows(pq.read_schema(next((roots.release / "data").rglob("*.parquet")))) == 256
    assert SC.scan_release(roots.release, private_root=None, workers=1).findings == []
    r2 = copy_release(site, "poison_batched")
    rel = r2.release
    files = sorted((rel / "data").rglob("*.parquet"))
    t_b = pq.read_table(files[1])
    clock = pa.array(10 ** 12 + np.arange(t_b.num_rows, dtype=np.int64) * 10 ** 8, pa.int64())
    rewrite(files[1], t_b.set_column(t_b.schema.get_field_index("seg"), "seg", clock))
    t_c = pq.read_table(files[2])
    ts = np.array(t_c["t_s"].to_numpy(), dtype=float, copy=True)
    ts[700] = ts[699]                                                          # a repeat deep inside the file (3rd batch)
    rewrite(files[2], t_c.set_column(t_c.schema.get_field_index("t_s"), "t_s", pa.array(ts, pa.float64())))
    write_checksums(r2, workers=1, use_cache=False)
    rep = SC.scan_release(rel, private_root=None, workers=1)
    kinds = sorted((f.gate, f.rule, f.column) for f in rep.findings)
    assert kinds == [("G5", "MAGNITUDE", "seg"), ("G5", "T_S", "t_s")], rep.counts()
