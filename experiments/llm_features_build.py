#!/usr/bin/env python3
"""Turn the feature-discovery workflow output into the `experiments/llm_features/` package.

Input: proposals.json = {lenses, features: [{name, lens, keep, reason, code, rationale, hypothesis, channels, finite_frac,
auroc_bev_vs_ice, spearman_footprint, ...}], audit: {verdicts: [{name, verdict, reason}], gaps, summary}} written by the
orchestrator from the workflow result. Output: one module per feature (`f_<name>.py`, the agent's code verbatim with a
provenance header), `_manifest.json` (metadata incl. the audit verdict) and `__init__.py` (registry FEATURES name -> fn).
Features that the verifier dropped or the auditor excluded are listed in the manifest but not registered. Finally every
registered function is executed on the first windows of the training sample; a function that raises is unregistered.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "experiments" / "llm_features"
from vtt import config as C  # noqa: E402

LLM_FEAT_DIR = Path(C.DATA_ROOT) / "experiments" / "llm_feat"      # agents' sample, notes and proposals (never released)
DEFAULT_IN = LLM_FEAT_DIR / "proposals.json"
DEFAULT_SAMPLE = LLM_FEAT_DIR / "train_sample.npz"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9_])" + "AD" + "AS" + r"(?![A-Za-z0-9_])")      # the public wording is "DAS"
_LOCAL_PATH_RE = re.compile(r"/(?:data|home|mnt|tmp)/[^\s`)\]\"]*/")


def wording(text) -> str:
    """Agent prose normalised for the public repository: the driver-assistance acronym becomes DAS, local paths are cut."""
    t = _ACRONYM_RE.sub("DAS", str(text))
    return _LOCAL_PATH_RE.sub("", t)


def function_name(code: str) -> str | None:
    m = re.search(r"^def\s+(feat_[A-Za-z0-9_]+)\s*\(", code, flags=re.M)
    return m.group(1) if m else None


def header(f: dict, verdict: dict) -> str:
    lines = [f"# Feature {f['name']} (lens: {f.get('lens', '?')}) - proposed and verified by LLM agents, code kept verbatim.",
             f"# Rationale: {f.get('rationale', '').strip()}",
             f"# Hypothesis: {f.get('hypothesis', '').strip()}",
             f"# Channels: {', '.join(f.get('channels', []))}",
             f"# Training-sample statistics: finite {f.get('finite_frac', float('nan')):.2f}, AUROC BEV-vs-ICE "
             f"{f.get('auroc_bev_vs_ice', float('nan')):.3f}, Spearman(footprint) {f.get('spearman_footprint', float('nan')):.3f}",
             f"# Audit: {verdict.get('verdict', 'keep')} - {verdict.get('reason', '').strip()}"]
    return "\n".join(wording(line).replace("\n", " ") for line in lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposals", type=Path, default=DEFAULT_IN)
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--n-test", type=int, default=3000, help="training-sample windows used by the self-test")
    ap.add_argument("--selftest-min-finite", type=float, default=0.05, help="a function finite in fewer windows than this is treated as broken")
    ap.add_argument("--readmit-min-finite", type=float, default=0.10,
                    help="re-admit features the verifiers dropped only for coverage when their finite fraction is at least this "
                         "(gradient boosting treats NaN as 'regime absent'); 1.0 disables re-admission")
    a = ap.parse_args(argv)
    j = json.loads(a.proposals.read_text())
    verdicts = {v["name"]: v for v in (j.get("audit") or {}).get("verdicts", [])}
    cov_re = re.compile(r"coverage|finite|fraction|0\.4", re.I)
    bad_re = re.compile(r"\bbug\b|exception|violat|duplicate|zero variance|constant|not fixable|forbidden|could not be made", re.I)
    n_readmit = 0
    for f in j["features"]:
        if not f.get("keep", False) and (f.get("finite_frac") or 0) >= a.readmit_min_finite and cov_re.search(f.get("reason", "")) \
                and not bad_re.search(f.get("reason", "")):
            f["keep"] = True
            f["reason"] = f"re-admitted (coverage-only drop, finite {f.get('finite_frac'):.2f} >= {a.readmit_min_finite}): " + f.get("reason", "")
            n_readmit += 1
    PKG.mkdir(parents=True, exist_ok=True)
    for old in PKG.glob("f_*.py"):
        old.unlink()
    manifest, registered, seen = [], [], set()
    for f in j["features"]:
        name = str(f["name"]).strip()
        v = verdicts.get(name, {})
        status = "kept"
        if not f.get("keep", False):
            status = "dropped_by_verifier"
        elif v.get("verdict") == "exclude":
            status = "excluded_by_audit"
        elif not NAME_RE.match(name) or name in seen:
            status = "bad_or_duplicate_name"
        fn = function_name(f.get("code", ""))
        if status == "kept" and fn is None:
            status = "no_function_in_code"
        entry = dict(name=name, lens=f.get("lens"), status=status, reason=wording(f.get("reason", "")), rationale=wording(f.get("rationale", "")),
                     hypothesis=wording(f.get("hypothesis", "")), channels=f.get("channels", []), finite_frac=f.get("finite_frac"),
                     auroc_bev_vs_ice=f.get("auroc_bev_vs_ice"), spearman_footprint=f.get("spearman_footprint"),
                     median_small=f.get("median_small"), median_mid=f.get("median_mid"), median_large=f.get("median_large"),
                     median_bev=f.get("median_bev"), median_ice=f.get("median_ice"), fixed=f.get("fixed"),
                     audit_verdict=v.get("verdict", "keep" if status == "kept" else None), audit_reason=wording(v.get("reason", "")), function=fn)
        manifest.append(entry)
        if status != "kept":
            continue
        seen.add(name)
        code = wording(f["code"]).strip() + "\n"
        if "import numpy as np" not in code:
            code = "import numpy as np\n\n" + code
        (PKG / f"f_{name}.py").write_text(header(f, v) + "\n" + code)
        registered.append((name, fn))
    init = ['"""Physics-based window features proposed by LLM agents (see experiments/LLM_FEATURES.md).',
            "",
            "Every module f_<name>.py holds one feature function; FEATURES maps the feature name to it. Built by",
            "experiments/llm_features_build.py from the verified proposals; do not edit the feature modules by hand.",
            '"""', "from __future__ import annotations", "", "import json", "from importlib import import_module", "from pathlib import Path", "",
            "MANIFEST = json.loads((Path(__file__).parent / '_manifest.json').read_text())", "FEATURES = {}",
            "for _m in MANIFEST:", "    if _m['status'] != 'kept':", "        continue",
            "    _mod = import_module(f\"experiments.llm_features.f_{_m['name']}\")", "    FEATURES[_m['name']] = getattr(_mod, _m['function'])",
            "", "NAMES = list(FEATURES)", ""]
    (PKG / "__init__.py").write_text("\n".join(init))
    (PKG / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    # self-test on training windows
    z = np.load(a.sample, allow_pickle=True)
    X, channels = np.asarray(z["X"][: a.n_test], dtype=np.float32), [str(c) for c in z["channels"]]
    sys.path.insert(0, str(REPO))
    if "experiments.llm_features" in sys.modules:
        del sys.modules["experiments.llm_features"]
    pkg = importlib.import_module("experiments.llm_features")
    bad = []
    for name, fn in pkg.FEATURES.items():
        vals = []
        try:
            for i in range(len(X)):
                w = {c: X[i, :, k] for k, c in enumerate(channels)}
                vals.append(float(fn(w)))
        except Exception as e:                  # noqa: BLE001
            bad.append((name, type(e).__name__))
            continue
        vals = np.asarray(vals, dtype=np.float64)
        fin = np.isfinite(vals).mean()
        if fin < a.selftest_min_finite:
            bad.append((name, f"finite {fin:.2f}"))
    if bad:
        for name, why in bad:
            for m in manifest:
                if m["name"] == name:
                    m["status"] = f"failed_selftest: {why}"
            (PKG / f"f_{name}.py").unlink(missing_ok=True)
        (PKG / "_manifest.json").write_text(json.dumps(manifest, indent=1))
    n_kept = sum(1 for m in manifest if m["status"] == "kept")
    print(f"features total={len(manifest)} registered={n_kept} re-admitted={n_readmit} failed_selftest={len(bad)} "
          f"statuses={ {s: sum(1 for m in manifest if m['status'] == s) for s in sorted({m['status'] for m in manifest})} }")
    return 0 if n_kept else 2


if __name__ == "__main__":
    sys.exit(main())
