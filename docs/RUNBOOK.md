# VTT runbook (who runs what, in order)

All commands assume `source $R/env.sh` first (sets `$R`, `$D`, `$PY`, `$CPY`
and moves every cache off the root disk). "User" steps read the raw library or publish; "agent" steps are local and short.
Every script prints counts only and exits 2 when its acceptance check fails.

| step | who | command | typical time | output |
|---|---|---|---|---|
| S1 census + uid map | user | `nice -n 15 ionice -c3 $PY $R/scripts/s01_probe_uidmap.py --workers 8` | 2 min (5 Nov 2026: 100 s) | `private/uid_map.parquet`, `build/routes_all.csv`, `build/chunks/` |
| S2 extraction | user | `bash $R/scripts/s02_extract_all.sh` (`--status` = read-only plan) | ~1 h with the 4,843-route hardlink reuse | `work/signals/<uid>/`, `work/tables/segments.parquet`, `build/qc/s02_accept.md` |
| S2b label audit probe | user | `nice -n 15 ionice -c3 $PY $R/scripts/s02b_audit_probe.py --workers 8` | 20-40 min | `private/route_audit.parquet` |
| S3a platforms + specs + labels | agent | `$PY $R/scripts/s03_platforms.py && $PY $R/scripts/s03b_specs_fill.py && $PY $R/scripts/s03_labels.py && $PY $R/scripts/s03_inventory.py` | 1 min | `vtt/specs_data/*.csv`, `build/routes_labels.parquet`, `build/SIZE_INVENTORY.md` |
| S3c lock thresholds | agent (rule in `docs/size_classes_rationale.md`; the user may override) | `$PY $R/scripts/s03c_lock_thresholds.py --thresholds 4.43,4.88 --rule docs/size_classes_rationale.md --write-config` then `s03_labels.py`, `s05_build_drives.py --relabel`, `s06_windows.py`, `s07b_apply_exclusions.py`, `s07_qc.py`, `s08_splits.py` | 5 min | `docs/SIZE_CLASSES.md`, config thresholds + sha256 |
| S4 pseudonyms | agent | `$PY $R/scripts/s04_pseudonyms.py` | seconds | `build/drive_names.parquet`, `private/pseudonym_map.parquet` |
| S5 drive tables | agent | `nice -n 10 ionice -c3 $PY $R/scripts/s05_build_drives.py --workers 8` (`--relabel` refreshes only the label columns of the index after a label change; `--no-resume` rebuilds the tables) | 6 min (relabel: seconds) | `build/data/<vehicle>/<drive>.parquet`, `build/index/drives.parquet` |
| S6 windows | agent | `$PY $R/scripts/s06_windows.py` | minutes | `build/index/windows.parquet` |
| S7 QC | agent, user reviews | `$PY $R/scripts/s07_qc.py` | minutes | `build/qc/REVIEW.md` |
| S8 splits | agent | `$PY $R/scripts/s08_splits.py` (refuses until S3c is locked) | 1 min | `build/splits/`, split columns in the index tables |
| S9 package + gate | agent | `$PY $R/scripts/s09_package.py` then `$PY $R/scripts/s09b_gate.py pre-upload` | 15-30 min | `release/v1/`, `.gate_ok` |
| S10 upload | agent (authorised by the user) | `$CPY $R/scripts/s10_upload.py readme-validate && ... upload-data && ... readme && ... check` | 30-90 min | HF dataset repo |
| S11 GitHub | agent exports, user pushes over SSH | `$PY $R/scripts/s11_github_export.py` then `git push` from `$D/github_export` | minutes | GitHub repo |
| S13 experiments | agent (GPU) | `$PY $R/scripts/s13_export_tensors.py --workers 8` (10 min), then `$CPY -m experiments.features_baseline` and `$CPY -m experiments.run_all --models transformer,gru,resnet,mamba2` (hours) | hours | `work/tensors/`, `<data-root>/experiments/results/*.json`, `experiments/RESULTS.md` |

Re-running any step is safe: outputs are written atomically and resumed where that makes sense.
