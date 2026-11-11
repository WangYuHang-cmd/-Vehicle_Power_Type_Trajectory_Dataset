#!/usr/bin/env python3
"""Segment list of chosen platforms from the commaCarSegments index, restricted to an existing selection list.

Used for the Honda re-decode (`ccs_honda`): the budget-list segments of the nine platforms that the first decode lost
(`CCS_PIPELINE.md`, "Honda re-decode"), and for the 20-segment consistency check of `ccs_honda_check.py`. Reads
`database.json` (platform key -> segments), folds the legacy keys with opendbc MIGRATION exactly as `ccs_select.py` does
(`ccs_platforms.find_opendbc_root` / `load_opendbc_names`, no opendbc import), keeps the entries present in `--within`
(one `<dongle>/<route>/<seg>` per line) and writes them, sorted, to `--out`. With `--limit N` the list is drawn round-robin
over the platforms and their devices (one segment per device until N is reached), so a small check list spans platforms
and devices. Prints counts only; the lists carry device ids and stay on the data disk (`$EXT/ccs/`).

  source env.sh && $CPY experiments/transfer/ccs_honda_list.py --platforms HONDA_CIVIC HONDA_CIVIC_BOSCH HONDA_CIVIC_2022 \\
      HONDA_CRV_5G HONDA_FIT HONDA_CRV HONDA_HRV HONDA_HRV_3G HONDA_FREED --out $EXT/ccs/select_honda.txt
  $CPY experiments/transfer/ccs_honda_list.py --platforms HONDA_ODYSSEY HONDA_PILOT HONDA_ACCORD --limit 20 --out $EXT/ccs/ccs_honda_check.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.transfer import ccs_platforms as P  # noqa: E402

HONDA_LOST = ("HONDA_CIVIC", "HONDA_CIVIC_BOSCH", "HONDA_CIVIC_2022", "HONDA_CRV_5G", "HONDA_FIT", "HONDA_CRV", "HONDA_HRV", "HONDA_HRV_3G", "HONDA_FREED")


def entry_key(s: str) -> str:
    """'<dongle>/<route>/<seg>/s' (index spelling) -> '<dongle>/<route>/<seg>'."""
    parts = s.strip().strip("/").split("/")
    return "/".join(parts[:3])


def collect(index: Path, within: Path, platforms: list[str], opendbc_root: Path | None = None) -> dict[str, list[str]]:
    """{platform: sorted segment keys of the index that are in `within`}."""
    migration, _ = P.load_opendbc_names(P.find_opendbc_root(opendbc_root))
    want = set(platforms)
    keep = {ln.strip().rstrip("/") for ln in Path(within).read_text().splitlines() if ln.strip()}
    out: dict[str, set[str]] = defaultdict(set)
    for key, entries in json.loads(Path(index).read_text()).items():
        f = P.fold(key, migration)
        if f not in want:
            continue
        for e in entries:
            k = entry_key(e)
            if k in keep:
                out[f].add(k)
    return {p: sorted(out.get(p, ())) for p in platforms}


def round_robin(by_platform: dict[str, list[str]], limit: int) -> list[str]:
    """Up to `limit` keys: cycle over the platforms, inside a platform over its devices, one segment per device per round."""
    queues: dict[str, list[list[str]]] = {}
    for p, keys in by_platform.items():
        by_dev: dict[str, list[str]] = defaultdict(list)
        for k in keys:
            by_dev[k.split("/")[0]].append(k)
        queues[p] = [sorted(v) for _, v in sorted(by_dev.items())]
    picked: list[str] = []
    while len(picked) < limit and any(queues.values()):
        for p in list(queues):
            devs = queues[p]
            if not devs or len(picked) >= limit:
                continue
            dev = devs.pop(0)
            picked.append(dev.pop(0))
            if dev:
                devs.append(dev)
    return sorted(picked)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=P.CCS_DIR / "database.json", help="commaCarSegments index (platform key -> segments)")
    ap.add_argument("--within", type=Path, default=P.CCS_DIR / "select_budget.txt", help="only segments of this list (one <dongle>/<route>/<seg> per line)")
    ap.add_argument("--platforms", nargs="+", default=list(HONDA_LOST), help="folded opendbc platform names (default: the nine lost Honda platforms)")
    ap.add_argument("--limit", type=int, default=0, help="draw at most N segments round-robin over platforms and devices (0 = every segment)")
    ap.add_argument("--out", type=Path, required=True, help="output list (data disk only; carries device ids)")
    ap.add_argument("--opendbc-root", type=Path, default=None, help="opendbc source tree for MIGRATION (default: the decode venv)")
    a = ap.parse_args(argv)
    by_platform = collect(a.index, a.within, a.platforms, a.opendbc_root)
    keys = round_robin(by_platform, a.limit) if a.limit > 0 else sorted(k for v in by_platform.values() for k in v)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(keys) + ("\n" if keys else ""))
    chosen = set(keys)
    for p in a.platforms:
        ks = [k for k in by_platform[p] if k in chosen]
        print(f"  {p:22s} in list {len(by_platform[p]):5d} segments / {len({k.split('/')[0] for k in by_platform[p]}):3d} devices"
              f" -> written {len(ks):5d} / {len({k.split('/')[0] for k in ks}):3d}")
    print(f"[ccs_honda_list] {len(keys)} segments, {len({k.split('/')[0] for k in keys})} devices, "
          f"{len({'/'.join(k.split('/')[:2]) for k in keys})} routes -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
