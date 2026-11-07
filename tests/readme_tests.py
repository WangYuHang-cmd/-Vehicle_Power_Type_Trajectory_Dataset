"""Dataset-card tests: the template carries no digits, every placeholder is produced by the packager, the rendered card
passes the text rules, and the driver-assistance acronym appears only inside the bibtex fence."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.synth_build import make_build
from vtt import config as C
from vtt import readme as RM
from vtt import schema as S
from vtt.gate import rules as R
from vtt.package import compute_readme_numbers, load_build, render_numbers
from vtt.paths import P


@pytest.fixture(scope="module")
def numbers():
    P.TMP.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vtt_readme_", dir=P.TMP))
    make_build(root / "build")
    drives, windows, splits_json = load_build(root / "build")
    shard_counts = {"n_shards": 3, "n_windows": 12, "bytes": 0, "per_split": [{"split": "train", "n_files": 1, "windows": 4, "bytes": 0}]}
    nums = compute_readme_numbers(drives, windows, splits_json, shard_counts, 0, "split_platform", R.load_allow()["email"])
    yield nums
    if not os.environ.get("VTT_KEEP_TMP"):
        shutil.rmtree(root, ignore_errors=True)


def test_template_has_no_digits():
    assert RM.template_digits(RM.template_text()) == []


def test_every_placeholder_is_produced(numbers):
    text = RM.template_text()
    missing = [p for p in RM.placeholders(text) if p not in render_numbers(numbers)]
    assert missing == []
    rendered = RM.render(text, render_numbers(numbers))
    assert "{{" not in rendered and "}}" not in rendered
    assert RM.fmt(numbers["n_drives"]) in rendered and str(C.NOMINAL_HZ) + " Hz" in rendered


def test_render_rejects_missing_keys():
    with pytest.raises(KeyError):
        RM.render("a {{nope}} b", {})


def test_rendered_card_passes_text_rules_and_fence_rule(numbers):
    rendered = RM.render(RM.template_text(), render_numbers(numbers))
    assert R.scan_text(rendered, readme=True, allowed_email=R.load_allow()["email"]) == {}
    assert R.hits_acronym(rendered, readme=True) == 0
    assert R.hits_acronym(rendered, readme=False) > 0                        # the word lives inside the bibtex fence only
    assert "driving automation system (DAS)" in rendered and "arXiv:2603.06986" in rendered


def test_generated_blocks_cover_the_schema():
    table = RM.field_table()
    for name, *_ in S.DRIVE_COLUMNS:
        assert f"`{name}`" in table
    official = RM.official_inputs_table()
    for name in C.OFFICIAL_INPUTS:
        assert f"`{name}`" in official
    nuisance = RM.nuisance_columns_list()
    assert "`steer_torque_raw`" in nuisance and "`lead_source`" in nuisance and "`v_mps`" not in nuisance
    blocks = RM.blocks()
    assert {"field_table", "official_inputs", "nuisance_columns", "drives_index_table", "windows_index_table", "vehicles_table", "platforms_table"} <= set(blocks)


def test_fmt_and_license_text():
    assert RM.fmt(1234567) == "1,234,567" and RM.fmt(3.14159) == "3.1" and RM.fmt(None) == "n/a" and RM.fmt(True) == "yes" and RM.fmt("x") == "x"
    text = RM.license_text("DatasetName", "https://example.org/d")
    assert text.startswith("DatasetName") and "Attribution 4.0 International Public License" in text and "Section 8" in text
    assert R.scan_text(text) == {}


def test_citation_token_appears_only_inside_the_single_bibtex_fence(numbers):
    rendered = RM.render(RM.template_text(), render_numbers(numbers))
    fences = R.BIBTEX_FENCE.findall(rendered)
    assert len(fences) == 1
    outside = R.BIBTEX_FENCE.sub(" ", rendered)
    assert R.CITATION_TOKEN.findall(outside) == [] and R.ACRONYM_WORD.findall(outside) == []
    assert len(R.CITATION_TOKEN.findall(fences[0])) >= 1
    assert "corpus" not in rendered.split("## Citation")[1].split("```")[0]   # no provenance sentence before the fence
