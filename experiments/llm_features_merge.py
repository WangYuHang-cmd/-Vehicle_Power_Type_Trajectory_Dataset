#!/usr/bin/env python3
"""Merge the feature-discovery workflow journals into one proposals.json for experiments/llm_features_build.py.

Round 1 journal: six propose/verify pairs plus one audit (verdicts for the features the verifiers kept).
Round 2 journal: one gap-filler agent (new, already verified features) plus one audit over the re-admitted features
(coverage-only drops of round 1) and the new ones. Output: {lenses, features, audit:{verdicts, gaps, summaries}} where every
feature carries its lens and the audit verdict list covers rounds 1 and 2 (a later verdict for the same name wins).
Nothing here reads data; it only re-shapes the agents' JSON returns.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402


def read_journal(path: Path) -> list[tuple[str, dict]]:
    """[(label, result dict)] for every finished agent of a workflow journal."""
    label: dict[str, str] = {}
    out: list[tuple[str, dict]] = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "started":
            label[r["agentId"]] = r.get("label", "")
        elif r.get("type") == "result" and isinstance(r.get("result"), dict):
            out.append((label.get(r.get("agentId"), ""), r["result"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round1", type=Path, required=True, help="journal.jsonl of the propose/verify/audit workflow")
    ap.add_argument("--round2", type=Path, default=None, help="journal.jsonl of the gap-filler / audit-2 workflow")
    ap.add_argument("--out", type=Path, default=Path(C.DATA_ROOT) / "experiments" / "llm_feat" / "proposals.json")
    a = ap.parse_args(argv)
    lenses, features, verdicts, gaps, summaries = [], [], {}, [], []
    for lab, res in read_journal(a.round1):
        if lab.startswith("verify:"):
            lens = res.get("lens", lab[len("verify:"):])
            lenses.append(dict(lens=lens, summary=res.get("summary", ""), duplicates_dropped=res.get("duplicates_dropped", [])))
            for f in res.get("features", []):
                features.append({**f, "lens": lens, "round": 1})
        elif lab == "audit":
            for v in res.get("verdicts", []):
                verdicts[v["name"]] = {**v, "round": 1}
            gaps += res.get("gaps", []) or []
            summaries.append("Audit 1: " + res.get("summary", ""))
    if a.round2 and a.round2.exists():
        for lab, res in read_journal(a.round2):
            if lab.startswith("fill:"):
                lens = res.get("lens", "gap_fillers")
                lenses.append(dict(lens=lens, summary=res.get("summary", ""), duplicates_dropped=[]))
                for f in res.get("features", []):
                    features.append({**f, "lens": lens, "round": 2})
            elif lab == "audit2":
                for v in res.get("verdicts", []):
                    verdicts[v["name"]] = {**v, "round": 2}
                summaries.append("Audit 2: " + res.get("summary", ""))
    names = [f["name"] for f in features]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        print(f"warning: {len(dup)} duplicate feature names across lenses; the first occurrence is kept")
        seen, uniq = set(), []
        for f in features:
            if f["name"] in seen:
                continue
            seen.add(f["name"])
            uniq.append(f)
        features = uniq
    out = dict(lenses=lenses, features=features,
               audit=dict(verdicts=list(verdicts.values()), gaps=gaps, summary="\n\n".join(summaries)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    n_round2 = sum(1 for f in features if f.get("round") == 2)
    print(f"features={len(features)} (round 2: {n_round2}) kept_by_verifiers={sum(bool(f.get('keep')) for f in features)} "
          f"verdicts={len(verdicts)} lenses={len(lenses)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
