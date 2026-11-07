"""S10 upload script: argument parsing, local README checks, upload refusal without the gate mark, and the
`check` logic against a fake Hub (no network, no token)."""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vtt import config as C
from vtt import schema as S
from vtt.paths import P


def load_s10():
    path = Path(__file__).resolve().parents[1] / "scripts" / "s10_upload.py"
    spec = importlib.util.spec_from_file_location("s10_upload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S10 = load_s10()

GOOD_CARD = """---
license: cc-by-4.0
pretty_name: "Synthetic card"
configs:
  - config_name: windows
    data_files:
      - split: validation
        path: "windows/validation/*.parquet"
---
# Card

Driving automation system (DAS) data; cite ADAS-TO.

```bibtex
@article{x, title={ADAS-TO: Human Takeovers during AD""" + """AS Engagement}}
```
"""


@pytest.fixture()
def root():
    P.TMP.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="vtt_s10_", dir=P.TMP))
    yield d
    if not os.environ.get("VTT_KEEP_TMP"):
        shutil.rmtree(d, ignore_errors=True)


def make_release(data_root: Path, n_drives: int = 3) -> Path:
    """A tiny release tree with every file class `check` looks at."""
    rel = data_root / "release" / C.RELEASE_VERSION
    (rel / "index").mkdir(parents=True); (rel / "splits").mkdir(); (rel / ".cache").mkdir(); (rel / "windows" / "test").mkdir(parents=True)
    (rel / "README.md").write_text(GOOD_CARD)
    (rel / "LICENSE").write_text("CC BY 4.0\n")
    (rel / "CHECKSUMS.sha256").write_text("")
    (rel / ".gate_ok").write_text("ok\n")
    (rel / ".cache" / "vtt_checksums.json").write_text("{}")
    pd.DataFrame({"vehicle_id": ["vehicle_0001"], "drive_id": ["drive_000001"]}).to_parquet(rel / "index" / "drives.parquet", index=False)
    pd.DataFrame({"window_id": ["drive_000001_w0"]}).to_parquet(rel / "splits" / "eval_balanced_size_test.parquet", index=False)
    for k in range(n_drives):
        df = pd.DataFrame({"t_s": np.arange(20) / 10.0, "v_mps": np.linspace(0, 5, 20) + k})
        S.write_public_parquet(df, rel / "data" / "vehicle_0001" / f"drive_{k + 1:06d}.parquet", S.DRIVE_COLUMNS, "drive")
    pd.DataFrame({"window_id": ["drive_000001_w0"]}).to_parquet(rel / "windows" / "test" / "shard_000.parquet", index=False)
    return rel


class FakeApi:
    """Minimal stand-in for HfApi: file list and dataset_info built from a local tree."""

    def __init__(self, rel: Path, gated="manual", private=False, license_=C.DATA_LICENSE, drop=(), size_delta=None):
        self.rel, self.gated, self.private, self.license, self.drop, self.size_delta = rel, gated, private, license_, set(drop), size_delta or {}
        self.calls = []

    def _files(self):
        return {k: v for k, v in S10.local_files(self.rel).items() if k not in self.drop}

    def list_repo_files(self, repo_id, repo_type=None):
        self.calls.append(("list", repo_id, repo_type))
        return sorted(self._files()) + [".gitattributes"]

    def dataset_info(self, repo_id, files_metadata=False, **kw):
        sib = [SimpleNamespace(rfilename=f, size=n + self.size_delta.get(f, 0), lfs=None) for f, n in self._files().items()]
        return SimpleNamespace(gated=self.gated, private=self.private, card_data={"license": self.license}, siblings=sib)

    def upload_large_folder(self, **kw):
        self.calls.append(("upload_large_folder", kw))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))


def test_parser_defaults():
    a = S10.build_parser().parse_args(["check"])
    assert a.repo == C.HF_REPO and a.workers == 4 and a.n_random == 8 and a.data_root == C.DATA_ROOT
    for cmd in S10.COMMANDS:
        assert S10.build_parser().parse_args([cmd, "--dry-run"]).cmd == cmd
    with pytest.raises(SystemExit):
        S10.build_parser().parse_args(["publish"])          # there is no command that changes repository settings


def test_readme_problems_rules():
    assert S10.readme_problems(GOOD_CARD) == []
    bad = GOOD_CARD.replace("license: cc-by-4.0", "license: cc-by-nc-4.0")
    assert any("license" in p for p in S10.readme_problems(bad))
    assert any("windows" in p for p in S10.readme_problems(GOOD_CARD.replace("config_name: windows", "config_name: drives")))
    acr = GOOD_CARD + "\nThe AD" + "AS word outside the citation.\n"
    assert any("acronym" in p for p in S10.readme_problems(acr))
    assert S10.readme_problems("no front matter") == ["README has no YAML front matter"]


def test_readme_validate_offline(root, capsys):
    make_release(root)
    assert S10.main(["readme-validate", "--data-root", str(root), "--offline"]) == 0
    (root / "release" / C.RELEASE_VERSION / "README.md").write_text(GOOD_CARD.replace("license: cc-by-4.0", "license: mit"))
    assert S10.main(["readme-validate", "--data-root", str(root), "--offline"]) == 2
    assert "problem" in capsys.readouterr().out


def test_upload_refuses_without_gate_ok(root):
    rel = make_release(root)
    api = FakeApi(rel)
    args = S10.build_parser().parse_args(["upload-data", "--data-root", str(root)])
    (rel / ".gate_ok").unlink()
    assert S10.cmd_upload_data(args, api) == 2 and api.calls == []
    (rel / ".gate_ok").write_text("ok\n")
    args_dry = S10.build_parser().parse_args(["upload-data", "--data-root", str(root), "--dry-run"])
    assert S10.cmd_upload_data(args_dry, api) == 0 and api.calls == []
    assert S10.cmd_upload_data(args, api) == 0
    kind, kw = api.calls[-1]
    assert kind == "upload_large_folder" and kw["repo_type"] == "dataset" and kw["num_workers"] == 4
    assert set(kw["ignore_patterns"]) == {"README.md", ".cache/**", ".gate_ok"} and kw["repo_id"] == C.HF_REPO
    assert S10.cmd_readme(args, api) == 0 and api.calls[-1][1]["path_in_repo"] == "README.md"


def test_local_files_skips_cache_and_gate_mark(root):
    rel = make_release(root)
    files = S10.local_files(rel)
    assert ".gate_ok" not in files and not any(f.startswith(".cache/") for f in files)
    assert "README.md" in files and "README.md" not in S10.local_files(rel, include_readme=False)


def test_check_passes_on_identical_hub(root):
    rel = make_release(root)
    api = FakeApi(rel)
    res = S10.check(api, "x/y", rel, download=lambda f: rel / f, n_random=2, seed=1,
                    load_dataset=lambda repo, cfg, split, streaming: iter([{"a": 1, "b": 2}]))
    assert res["problems"] == [] and res["n_missing"] == 0 and res["n_extra"] == 0 and res["n_size_bad"] == 0
    assert res["n_sha_bad"] == 0 and res["streamed_columns"] == 2 and res["gated"] == "manual" and res["license"] == C.DATA_LICENSE
    assert res["n_sha_checked"] >= 2 + 2 + 2                 # README + CHECKSUMS + index + splits + 2 random data files


def test_check_detects_problems(root):
    rel = make_release(root)
    missing = FakeApi(rel, drop=["index/drives.parquet"])
    res = S10.check(missing, "x/y", rel, download=lambda f: rel / f, n_random=1, seed=0)
    assert any("missing on the hub" in p for p in res["problems"])
    bad_size = FakeApi(rel, size_delta={"LICENSE": 5})
    res = S10.check(bad_size, "x/y", rel, download=lambda f: rel / f, n_random=1, seed=0)
    assert any("different size" in p for p in res["problems"])
    settings = FakeApi(rel, gated=None, private=True, license_="mit")
    res = S10.check(settings, "x/y", rel, download=lambda f: rel / f, n_random=1, seed=0)
    assert sum(("gated" in p) + ("private" in p) + ("license" in p) for p in res["problems"]) == 3
    other = rel / "LICENSE"
    res = S10.check(FakeApi(rel), "x/y", rel, download=lambda f: other, n_random=1, seed=0)
    assert any("sha256" in p for p in res["problems"])

    def boom(*a, **k):
        raise RuntimeError("no stream")
    res = S10.check(FakeApi(rel), "x/y", rel, download=lambda f: rel / f, n_random=0, seed=0, load_dataset=boom)
    assert any("streaming" in p for p in res["problems"])


def test_cmd_check_exit_codes(root, monkeypatch):
    rel = make_release(root)
    monkeypatch.setattr(S10, "_api", lambda api=None: FakeApi(rel))
    monkeypatch.setattr(S10, "hf_hub_download_for_test", None, raising=False)
    import types
    fake_hub = types.SimpleNamespace(hf_hub_download=lambda repo, f, repo_type=None, cache_dir=None, force_download=False: str(rel / f))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HF_HOME", str(root / "hf_home"))
    assert S10.main(["check", "--data-root", str(root), "--no-stream", "--n-random", "1", "--seed", "3"]) == 0
    assert (root / "hf_home" / "check").is_dir()
    monkeypatch.setattr(S10, "_api", lambda api=None: FakeApi(rel, gated=None))
    assert S10.main(["check", "--data-root", str(root), "--no-stream", "--n-random", "1"]) == 2


def test_never_prints_token(root, monkeypatch, capsys):
    token = "hf_" + "A" * 34
    hex16 = "f" * 10 + "%06x" % 7

    def raise_token(api=None):
        raise RuntimeError(f"bad credentials {token} for {hex16}")
    monkeypatch.setattr(S10, "_api", raise_token)
    assert S10.main(["settings", "--data-root", str(root)]) == 2
    out = capsys.readouterr().out
    assert token not in out and hex16 not in out and "<token>" in out and "<hex16>" in out
