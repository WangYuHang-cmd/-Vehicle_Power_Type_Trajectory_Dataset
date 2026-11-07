#!/usr/bin/env python3
"""S10: publish release/<version> to the Hugging Face dataset repository (run with the miniconda python, $CPY).

Subcommands, in the order they are meant to run:
  readme-validate  local checks of release/<version>/README.md (front matter parses, license, configs, DAS wording)
                   and, unless --offline, the Hub's card validation (DatasetCard.validate posts the card text only)
  upload-data      upload_large_folder of the release tree (data/, index/, splits/, schema/, CHECKSUMS, LICENSE),
                   skipping README.md, .cache/** and .gate_ok; refuses to run when .gate_ok is missing
  readme           upload README.md last (so a card problem never stalls the bulk upload)
  check            repository file set == local set + .gitattributes; sizes from dataset_info(files_metadata=True);
                   sha256 of README, CHECKSUMS, every index/ and splits/ file and N random data files downloaded
                   into $HF_HOME/check; gated == 'manual', private == False, card license == cc-by-4.0; one streamed
                   row of the 'windows' config (validation split) when `datasets` is importable
  settings         print gated / private / license (never changes anything; the repository is never made private)
Prints counts and verdicts only; never prints a token. HF_HOME / HF_XET_CACHE come from the environment (env.sh).
Exit 2 on a failed check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from vtt import config as C
from vtt.private import mask

IGNORE_PATTERNS = ["README.md", ".cache/**", ".gate_ok"]
COMMIT_SIZE_SCALE = [20, 50, 75, 100]
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{10,}")
ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9_])" + "AD" + "AS" + r"(?![A-Za-z0-9_])", re.I)
BIBTEX_RE = re.compile(r"```bibtex[ \t]*\n.*?\n```", re.S)
REQUIRED_CARD_KEYS = ("license", "pretty_name", "configs")


def scrub(text) -> str:
    """Printable form of any message: identifier masking plus Hub tokens blanked."""
    return TOKEN_RE.sub("<token>", mask(str(text)))


def release_root(data_root: Path, override: Path | None = None) -> Path:
    return Path(override) if override else Path(data_root) / "release" / C.RELEASE_VERSION


def hf_home(data_root: Path) -> Path:
    return Path(os.environ.get("HF_HOME") or (Path(data_root) / "hf_home"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_files(root: Path, include_readme: bool = True) -> dict[str, int]:
    """Relative POSIX path -> size of every release file except .cache/** and .gate_ok (README optional)."""
    out: dict[str, int] = {}
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".cache/") or rel == ".gate_ok" or (rel == "README.md" and not include_readme):
            continue
        out[rel] = p.stat().st_size
    return out


# ------------------------------------------------------------------------------------------------ README
def parse_front_matter(text: str) -> dict:
    """YAML front matter of a dataset card as a dict (ValueError when absent or unparsable)."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise ValueError("README has no YAML front matter")
    import yaml                                           # noqa: WPS433 - pyyaml is a declared dependency
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        raise ValueError("front matter is not a mapping")
    return data


def readme_problems(text: str) -> list[str]:
    """Local card checks: front matter keys, license, a 'windows' config, the acronym rule (DAS wording)."""
    problems: list[str] = []
    try:
        fm = parse_front_matter(text)
    except ValueError as exc:
        return [str(exc)]
    for k in REQUIRED_CARD_KEYS:
        if k not in fm:
            problems.append(f"front matter lacks '{k}'")
    if fm.get("license") != C.DATA_LICENSE:
        problems.append(f"license is not {C.DATA_LICENSE}")
    names = [c.get("config_name") for c in fm.get("configs", []) if isinstance(c, dict)]
    if "windows" not in names:
        problems.append("no 'windows' config in the front matter")
    body = BIBTEX_RE.sub(" ", text, count=1).replace("ADAS-TO", " ")
    n_acr = len(ACRONYM_RE.findall(body))
    if n_acr:
        problems.append(f"driver-assistance acronym used {n_acr} time(s) outside the citation")
    return problems


def cmd_readme_validate(a, api=None) -> int:
    readme = release_root(a.data_root, a.release_root) / "README.md"
    if not readme.exists():
        print("README.md missing in the release tree")
        return 2
    problems = readme_problems(readme.read_text())
    for p in problems:
        print("  problem:", scrub(p))
    if not a.offline:
        try:
            from huggingface_hub import DatasetCard
            DatasetCard.load(readme).validate(repo_type="dataset")
            print("hub card validation: OK")
        except Exception as exc:                            # noqa: BLE001 - reported, never re-raised with a token
            problems.append("hub validation failed")
            print("hub card validation FAILED:", type(exc).__name__, scrub(getattr(exc, "args", [""])[0])[:300])
    print(f"readme-validate: {len(problems)} problem(s)")
    return 0 if not problems else 2


# ------------------------------------------------------------------------------------------------ upload
def _api(api=None):
    if api is not None:
        return api
    from huggingface_hub import HfApi
    return HfApi()


def set_commit_scale() -> list[int] | None:
    """Cap the commit batch growth (regular blobs are base64-inlined per commit); no-op when the knob is absent."""
    try:
        import huggingface_hub._upload_large_folder as ulf
    except ImportError:
        return None
    if hasattr(ulf, "COMMIT_SIZE_SCALE"):
        ulf.COMMIT_SIZE_SCALE[:] = COMMIT_SIZE_SCALE
        return list(ulf.COMMIT_SIZE_SCALE)
    return None


def cmd_upload_data(a, api=None) -> int:
    rel = release_root(a.data_root, a.release_root)
    if not (rel / ".gate_ok").exists():
        print("refusing to upload: .gate_ok missing (run the pre-upload gate first)")
        return 2
    files = local_files(rel, include_readme=False)
    print(f"upload-data: {len(files)} files, {sum(files.values()) / 1e9:.2f} GB, workers={a.workers}, ignore={IGNORE_PATTERNS}")
    if a.dry_run:
        return 0
    scale = set_commit_scale()
    print("commit size scale:", scale)
    api = _api(api)
    t0 = time.time()
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(rel), num_workers=a.workers,
                            ignore_patterns=IGNORE_PATTERNS, print_report=True, print_report_every=300)
    print(f"upload_large_folder finished in {time.time() - t0:.0f} s")
    return 0


def cmd_readme(a, api=None) -> int:
    readme = release_root(a.data_root, a.release_root) / "README.md"
    if not readme.exists():
        print("README.md missing in the release tree")
        return 2
    if a.dry_run:
        print("readme: would upload README.md")
        return 0
    _api(api).upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=a.repo, repo_type="dataset",
                          commit_message="dataset card")
    print("readme: uploaded")
    return 0


# ------------------------------------------------------------------------------------------------ check
def card_license(info, readme: Path | None) -> str | None:
    """License from the Hub card data, else from the local README front matter."""
    cd = getattr(info, "card_data", None)
    lic = None
    if cd is not None:
        lic = cd.get("license") if hasattr(cd, "get") else getattr(cd, "license", None)
    if lic is None and readme is not None and readme.exists():
        try:
            lic = parse_front_matter(readme.read_text()).get("license")
        except ValueError:
            lic = None
    return lic


def pick_sample(files: dict[str, int], n_random: int, rng: random.Random) -> list[str]:
    """README + CHECKSUMS + every index/ and splits/ file + n_random data / window files."""
    fixed = [f for f in files if f in ("README.md", "CHECKSUMS.sha256") or f.startswith(("index/", "splits/"))]
    pool = sorted(f for f in files if f.startswith(("data/", "windows/")))
    return fixed + rng.sample(pool, min(n_random, len(pool)))


def check(api, repo: str, rel: Path, download, n_random: int = 8, seed: int | None = None, load_dataset=None) -> dict:
    """All verifications of the published repository against the local release tree; returns counts + problems.
    `download(rel_path) -> local path` fetches one repository file; `load_dataset(repo, config, split)` streams."""
    rng = random.Random(seed)
    problems: list[str] = []
    local = local_files(rel)
    hub = set(api.list_repo_files(repo, repo_type="dataset"))
    expected = set(local) | {".gitattributes"}
    missing, extra = sorted(expected - hub), sorted(hub - expected)
    if missing:
        problems.append(f"{len(missing)} local files missing on the hub")
    if extra:
        problems.append(f"{len(extra)} extra files on the hub")
    info = api.dataset_info(repo, files_metadata=True)
    sizes = {s.rfilename: (s.lfs.size if getattr(s, "lfs", None) and getattr(s.lfs, "size", None) else s.size)
             for s in (getattr(info, "siblings", None) or [])}
    size_bad = [f for f, n in local.items() if f in sizes and sizes[f] is not None and sizes[f] != n]
    size_unknown = sum(1 for f in local if sizes.get(f) is None)
    if size_bad:
        problems.append(f"{len(size_bad)} files with a different size on the hub")
    sample = pick_sample(local, n_random, rng)
    sha_bad = []
    for f in sample:
        if f not in hub:
            continue
        try:
            if sha256(Path(download(f))) != sha256(rel / f):
                sha_bad.append(f)
        except Exception as exc:                            # noqa: BLE001
            problems.append(f"download failed: {type(exc).__name__}")
            break
    if sha_bad:
        problems.append(f"{len(sha_bad)} sampled files differ (sha256)")
    gated, private = getattr(info, "gated", None), getattr(info, "private", None)
    lic = card_license(info, rel / "README.md")
    if gated != "manual":
        problems.append(f"gated is {gated!r}, expected 'manual'")
    if private is not False:
        problems.append(f"private is {private!r}, expected False")
    if lic != C.DATA_LICENSE:
        problems.append(f"card license is {lic!r}, expected {C.DATA_LICENSE!r}")
    streamed = None
    if load_dataset is not None:
        try:
            row = next(iter(load_dataset(repo, "windows", split="validation", streaming=True)))
            streamed = len(row) if hasattr(row, "__len__") else 1
        except Exception as exc:                            # noqa: BLE001
            problems.append(f"streaming load failed: {type(exc).__name__}")
    return dict(n_local=len(local), n_hub=len(hub), n_missing=len(missing), n_extra=len(extra), n_size_checked=len(local) - size_unknown,
                n_size_unknown=size_unknown, n_size_bad=len(size_bad), n_sha_checked=len(sample) - len([f for f in sample if f not in hub]),
                n_sha_bad=len(sha_bad), gated=gated, private=private, license=lic, streamed_columns=streamed, problems=problems)


def _datasets_loader():
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        return None


def cmd_check(a, api=None) -> int:
    rel = release_root(a.data_root, a.release_root)
    api = _api(api)
    cache = hf_home(a.data_root) / "check"
    cache.mkdir(parents=True, exist_ok=True)

    def download(rel_path: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(a.repo, rel_path, repo_type="dataset", cache_dir=str(cache), force_download=True)

    res = check(api, a.repo, rel, download, n_random=a.n_random, seed=a.seed, load_dataset=None if a.no_stream else _datasets_loader())
    for k, v in res.items():
        if k != "problems":
            print(f"  {k}: {v}")
    for p in res["problems"]:
        print("  problem:", scrub(p))
    print(f"check: {len(res['problems'])} problem(s)")
    return 0 if not res["problems"] else 2


def cmd_settings(a, api=None) -> int:
    info = _api(api).dataset_info(a.repo)
    print(json.dumps({"gated": getattr(info, "gated", None), "private": getattr(info, "private", None),
                      "license": card_license(info, release_root(a.data_root, a.release_root) / "README.md")}))
    return 0


COMMANDS = {"readme-validate": cmd_readme_validate, "upload-data": cmd_upload_data, "readme": cmd_readme,
            "check": cmd_check, "settings": cmd_settings}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--release-root", type=Path, default=None, help="default <data-root>/release/<version>")
    ap.add_argument("--repo", default=C.HF_REPO)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=8, help="random data files to sha256-verify in `check`")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--offline", action="store_true", help="readme-validate: skip the Hub card validation call")
    ap.add_argument("--no-stream", action="store_true", help="check: skip the streaming load_dataset probe")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    a = build_parser().parse_args(argv)
    try:
        return COMMANDS[a.cmd](a)
    except Exception as exc:                                # noqa: BLE001 - never let a traceback print a token
        print(f"{a.cmd} failed: {type(exc).__name__}: {scrub(exc)[:300]}")
        return 2


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)          # the Hub client leaves worker threads behind; a hard exit avoids the interpreter teardown crash
