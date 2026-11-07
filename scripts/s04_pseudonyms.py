#!/usr/bin/env python3
"""S4: identity clusters + random pseudonyms (vehicle_####, drive_######, group_####).

Reads  private/routes_resolved.parquet, build/routes_labels.parquet (platform), ROUTE_MASTER_CSV (in_house, read-only),
       private/identity_merges.csv, private/misfiled_dirs.json, the extractor's lab lists (run-time import).
Writes private/pseudonym_map.parquet (append-only), private/pseudonym_state.json, private/release_seed.txt,
       private/inhouse_groups.csv, build/drive_names.parquet.
Prints counts only; exit 2 when an invariant fails. Idempotent: a re-run with unchanged inputs changes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vtt import config as C
from vtt import pseudonyms as PS
from vtt.private import mask


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=C.DATA_ROOT)
    ap.add_argument("--route-master", type=Path, default=C.ROUTE_MASTER_CSV, help="OverView route_master.csv (in_house column)")
    ap.add_argument("--no-route-master", action="store_true",
                    help="accept a missing route_master table (lab devices then come from the extractor lists / folder names only)")
    ap.add_argument("--no-tocf", action="store_true", help="do not import the extractor's lab lists / folder re-keys (tests)")
    ap.add_argument("--allow-regroup", action="store_true", help="accept clusters that merged or split since the last run")
    ap.add_argument("--allow-missing-labels", action="store_true", help="use the raw fingerprint as platform when S3a has not run")
    ap.add_argument("--dry-run", action="store_true", help="compute and print counts, write nothing")
    a = ap.parse_args(argv)
    try:
        counts = PS.run(a.data_root, a.route_master, use_tocf=not a.no_tocf, allow_regroup=a.allow_regroup,
                        allow_missing_labels=a.allow_missing_labels, dry_run=a.dry_run,
                        require_route_master=not a.no_route_master)
    except (AssertionError, FileNotFoundError, ValueError, OverflowError) as e:
        print("s04 FAILED:", mask(str(e)))
        return 2
    printable = {k: v for k, v in counts.items() if k != "groups"}
    printable["groups"] = {k: v for k, v in counts["groups"].items() if k != "lab_group_key"}
    printable["groups"]["lab_group_present"] = bool(counts["groups"].get("lab_group_key"))
    print(mask(json.dumps(printable, indent=1, sort_keys=True, default=str)))
    print(f"s04 in-house devices: {counts['n_inhouse_devices']} ({counts['n_inhouse_devices_with_routes']} with routes); "
          f"explicit merges used/skipped: {counts['merges']['n_used']}/{counts['merges']['n_skipped']}; "
          f"vehicle ids retired: {counts.get('n_vehicle_ids_retired', 0)}")
    if counts["problems"]:
        print(f"s04 FAILED: {len(counts['problems'])} invariant(s) violated")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
