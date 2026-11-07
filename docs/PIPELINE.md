# VTT build pipeline: stages, scripts and file contracts

Approved plan: (internal design plan, not published) (authoritative for design decisions).
Code contracts: `vtt/config.py` (switches), `vtt/paths.py` (every path), `vtt/schema.py` (public tables),
`vtt/private.py` (private root, atomic writers, masking), `vtt/tocf_bridge.py` (reuse of the ADAS-TO-CF extractor).

Environment: `source env.sh`; build/tests run with `$PY` (openpilot venv: pyarrow 17, pandas 2.2, numpy 2.5, no scipy,
no `datasets`); upload and starter_kit run with `$CPY` (miniconda: huggingface_hub 0.36, datasets 4.8).
Nothing is written to `/` (root disk 98% full): temp files go to `$D/tmp`.
Hard rules for every script: never modify `$VTT_RAW_ROOT`; never print, log or store a 16-hex dongle id,
a route id, a VIN, a person or folder name outside `$D/private` (use `vtt.private.mask`); exit code 2 on failed
acceptance checks; print counts, not identifiers.

## Stage table
| stage | script | reads | writes |
|---|---|---|---|
| S1 | `scripts/s01_probe_uidmap.py` | raw library (listing + first qlog of ~450 new dirs), `private/seeds/{salt.txt,route_probe.parquet}` | `private/salt.txt`, `private/raw_manifest.csv`, `private/route_probe.parquet`, `private/uid_map.parquet`, `private/routes_resolved.parquet`, `private/identity_merges.csv`, `private/misfiled_dirs.json` (unresolved non-hex folders), `build/routes_all.csv`, `build/routes_smoke20.csv`, `build/chunks/routes_chunk_NN.csv` |
| S2 | `scripts/s02_extract_all.sh`, `scripts/s02c_segments_table.py`, `scripts/s02d_accept.py` | `build/chunks/*.csv`, `private/uid_map.parquet`, `$VTT_TOCF_DATA/v1/signals` (hardlink source) | `work/signals/<route_uid>/*.parquet + route_meta.json`, `work/tables/segments.parquet`, `build/qc/s02_accept.{csv,md}` |
| S2b | `scripts/s02b_audit_probe.py` | raw library first 3 qlogs per route, `private/uid_map.parquet` | `private/route_audit.parquet` |
| S3a | `scripts/s03_platforms.py`, `scripts/s03_labels.py`, `scripts/s03_inventory.py` | `private/routes_resolved.parquet`, `TAXONOMY_DIR/csv/routes_taxonomy.csv`, `ROUTE_MASTER_CSV`, `OPENDBC_ROOT` (or local fallback), `vtt/specs/*.csv` | `vtt/specs/fingerprint_to_platform.csv`, `vtt/specs/platform_specs.csv`, `build/routes_labels.parquet`, `private/labels_private.parquet`, `build/SIZE_INVENTORY.md` |
| S4 | `scripts/s04_pseudonyms.py` | `private/routes_resolved.parquet`, `build/routes_labels.parquet`, `ROUTE_MASTER_CSV` (in_house), tocf lab lists | `private/pseudonym_map.parquet`, `private/release_seed.txt`, `private/inhouse_groups.csv`, `build/drive_names.parquet` |
| S5 | `scripts/s05_build_drives.py` | `work/signals/<uid>`, `build/drive_names.parquet`, `build/routes_labels.parquet`, `build/routes_all.csv` | `build/data/<vehicle_id>/<drive_id>.parquet`, `build/index/drives.parquet`, `private/qc/drives_private_qc.parquet` |
| S6 | `scripts/s06_windows.py` | `build/data`, `build/index/drives.parquet` | `build/index/windows.parquet`, `build/index/human_only_windows.parquet` |
| S7 | `scripts/s07_qc.py` | drives, windows, `private/route_audit.parquet`, `build/label_overrides.csv` (optional) | `build/qc/drives_qc.csv`, `build/qc/REVIEW.md`, `private/qc/label_audit_private.csv`, `label_note`/`label_audit` written back into `build/index/drives.parquet` |
| S3c | (user decision) | `build/SIZE_INVENTORY.md` | `docs/SIZE_CLASSES.md`, `vtt/config.py` thresholds + sha256 |
| S8 | `scripts/s08_splits.py` | drives, windows, `private/inhouse_groups.csv`, `private/identity_merges.csv` | `build/splits/splits.json`, `build/splits/eval_balanced_{size,bev_ice}_{validation,test}.parquet`, `build/splits/train_uniform_{size,bev_ice}.parquet`, `build/index/split_summary.csv`; split columns written into drives + windows |
| S9 | `scripts/s09_package.py` | build/ | `release/v1/**`, `release/v1/CHECKSUMS.sha256`, `release/v1/.gate_ok` |
| S10 | `scripts/s10_upload.py` (`$CPY`) | `release/v1` | HF repo |
| S11 | `scripts/s11_github_export.py` | repo | `$D/github_export` (fresh git) |

## Intermediate file contracts
- `private/uid_map.parquet`: exactly `tocf.schema.PRIVATE_UID_MAP` columns (`route_uid, car_dir, dongle_dir, dongle_log, route_id, identity_uid, vin_hash`), one row per logged route (the chosen copy). Read by the extractor.
- `private/routes_resolved.parquet`: uid_map columns + `rel_path` (car_dir/dongle_dir/route_id of the chosen copy), `copies_rel` (JSON list of all copies), `n_copies`, `n_segments`, `n_qlog`, `n_rlog`, `version`, `device_type`, `car_fp`, `opLong`, `radarUnavailable`, `pcmCruise`, `probe_status`, `dup_rule` (`none|identical_copy|shared_route_id`), `is_long` (n_segments > config.LONG_ROUTE_SEGMENTS).
- `build/routes_all.csv` (public-safe, internal uids only): `route_uid, n_segments, n_qlog, n_rlog, n_copies, software_era, device_type, op_long, radar_unavailable, probe_status, is_long`.
- `build/routes_smoke20.csv`: subset of routes_all with an extra `stratum` column.
- `build/chunks/routes_chunk_NN.csv`: `route_uid` column (<= config.CHUNK_ROUTES rows); the last file is `routes_chunk_long.csv`.
- `work/signals/<route_uid>/`: extractor output (`tocf.schema.SIGNAL_COLUMNS`, `logMonoTime` int64 ns + `seg`), `route_meta.json` (`tocf.schema.ROUTE_META_KEYS`).
- `private/route_audit.parquet`: `route_uid, n_segments_read, ecus (JSON list), has_hybrid_ecu, has_transmission_ecu, transmission, mass_kg, wheelbase_m, steer_ratio, engine_rpm_pos_frac, engine_off_moving_frac, regen_any, charging_any, fuel_gauge_med, gear_values (JSON), yaw_rate_nonzero_frac, eps_torque_nonzero_frac, n_samples, err`.
- `vtt/specs/fingerprint_to_platform.csv`: `fingerprint, platform, method (exact|migration|suffix|manual), note` for every distinct `fingerprint` of routes_taxonomy.csv plus every `car_fp` seen in the probe (100% coverage test).
- `vtt/specs/platform_specs.csv`: `platform, brand, model, body, generation_note, wheelbase_mm, track_front_mm, track_rear_mm, footprint_m2, footprint_min_m2, footprint_max_m2, curb_mass_kg, mass_source, wheelbase_source, track_source, spec_confidence (A|B|C), source_urls (space separated), note`.
- `build/routes_labels.parquet`: `route_uid, platform, brand, model, body, size_code, powertrain_raw, powertrain, label_source, label_grade, label_note, region, software_era, wheelbase_mm, track_front_mm, track_rear_mm, footprint_m2, curb_mass_kg, spec_confidence, size_class` (size_class null until S3c locks thresholds). `private/labels_private.parquet` = same + `rel_path, dongle_log, fingerprint_raw, in_house, country`.
- `build/drive_names.parquet`: `route_uid, vehicle_id, drive_id, group_id` (no other identifier). `private/pseudonym_map.parquet`: + `dongle_log, platform, route_id, rel_path, vin_hash, assigned_release`.
- `build/data/<vehicle_id>/<drive_id>.parquet`: `vtt.schema.DRIVE_COLUMNS`, written with `vtt.schema.write_public_parquet(..., 'drive')`.
- `build/index/drives.parquet`: `vtt.schema.DRIVES_INDEX_COLUMNS` (`split_*`, `low_support_cell` null until S8; `label_note` filled by S7).
- `build/index/windows.parquet`: `vtt.schema.WINDOWS_INDEX_COLUMNS`.
- `build/splits/splits.json`: `{version, seed, config: {...}, cells: [{size_class, powertrain, split: {train:{hours,groups,platforms,windows}, ...}, supported, low_support, max_group_share}], groups: {group_id: {split_device, split_platform}}, heldout_platforms: [...], class_weights: {size: {...}, bev_ice: {...}}, sha256_size_classes}`.

## Contract notes
Appended by the stage builders when they deviate from or refine the contracts above (append only).

### S1 — `scripts/s01_probe_uidmap.py`, `vtt/manifest.py` (2026-11-05)
- `private/routes_resolved.parquet` carries, beyond the listed columns, `software_era`, `route_style`, `n_qcamera`,
  `n_seg_rlog_only` (exact number of segments with an rlog and no qlog, from one directory listing of the candidate
  routes), `altExp`, `expLongAvail`, `git_commit`, `dir_uid`, `bytes`. All private.
- `build/routes_all.csv`: `op_long` / `radar_unavailable` are tri-state (True / False / empty when the probe did not read
  carParams); `device_type` is `unknown` when absent; `probe_status` is one of ok | no_carparams | no_initdata | no_logs |
  error:<Type> | unprobed.
- `build/routes_smoke20.csv` `stratum` values: `nonhex_folder`, `misfiled_dir`, `date_style`, `rlog_only`, `multi_copy`,
  `dup_rule`, `long` (one route each when such a route exists), then `era_<software_era>_<device_type>` cells.
- `build/chunks/routes_chunk_long.csv` is always written (header-only when no route exceeds `LONG_ROUTE_SEGMENTS`); stale
  `routes_chunk_*.csv` files are deleted on every run, so the directory is exactly the current chunk set.
- `private/identity_merges.csv` (append-only, keyed by `dongle_log_a, dongle_log_b, route_id`): `dongle_log_a, dongle_log_b,
  route_id, rule, evidence, route_uid_a, route_uid_b, n_seg_a, n_seg_b, added_by, added_at`. `rule` is `shared_route_id`;
  `evidence` is `counter_id` (random-suffix id shared by two devices: strong), `date_id_same_car` or `date_id_other_car`
  (date-style ids have 1-s resolution; the last one may be a coincidence and S8 may weigh it).
- `private/misfiled_dirs.json` is the flat `{folder: dongle_log}` table read by `vtt.private.misfiled_dirs()`. S1 adds the
  entries it learns from the probe (a non-hex folder -> the id its logged routes carry; a 16-hex folder whose logged routes
  all name one other id) without overriding a manual 16-hex entry, and maps a non-hex folder that has NO logged id anywhere
  to ITSELF: that is the "needs a manual entry" marker (its routes are excluded from uid_map and counted). Per-folder detail
  (kind, status, counts) is in `private/misfiled_dirs_detail.json`. Consumers must ignore values that are not 16-hex.
- Probe cache: besides directories absent from the cache, cached rows with status `no_logs` whose directory now holds a log
  and rows with `error:*` are probed again (`--no-reprobe-stale` disables this).
- Duplicate rule detail: content identity = same segment-number set and same sha256 of the decompressed lowest-numbered log
  (qlog, else rlog). Among identical copies the kept one prefers folder == logged id, probe ok, a 16-hex folder, more
  qlog + qcamera files, more rlogs, then the lexically smaller path; it absorbs the dropped copy's `n_copies` / `copies_rel`.
- `build/logs/s01_summary.json` holds counts only (dirs, logged dirs, probe status, folders, dup rules, chunks, smoke strata,
  `problems`, `accepted`); S2 should refuse to start when `accepted` is false. Exit code 2 mirrors `accepted`.

## Contract notes
S5 (`vtt/signals.py`, `vtt/drives.py`, `scripts/s05_build_drives.py`), appended 2026-11-05:
- `build/index/drives.parquet.file` = `data/<vehicle_id>/<drive_id>.parquet`, relative to the build root (= release root layout).
- `n_segments` = distinct source segments that contributed carState samples to the drive (so `seg` runs 0..n_segments-1); the
  route's total segment count from route_meta (`n_seg`) is kept in the private QC frame as `n_seg_meta`. `n_seg_rlog_dec` /
  `n_seg_none` count route_meta `segments[].src_kind`.
- `pose_source` (index) is derived from the drive's own held samples (`livePose` / `liveLocationKalman` / `mixed` / `none`);
  route_meta's value is kept privately as `pose_source_meta`. `has_pose` = a pose stream exists and at least one yaw sample was held.
- `software_era` comes from `build/routes_labels.parquet` when the route has a label row, else from route_meta.version through
  `vtt.drives.software_era` (which defers to `vtt.manifest.software_era` when that module exists; the fallback maps 0.8/0.9/0.10/0.11,
  2025, 2026, everything else -> `other`).
- `v_set_mps`: first valid of carState.vCruise, carState.vCruiseCluster, held controlsState.vCruise/vCruiseCluster, held
  selfdriveState.vCruise/vCruiseCluster (km/h on the wire -> m/s; 0 and the 255 km/h sentinel are 'unset'), then
  carState.cruiseState.speed (already m/s); a value above 45 m/s is 'unset' (`vtt.drives.V_SET_MAX_MPS`, = tocf V_SET_MAX_MS).
- `qc_lead_age_s` / `qc_yaw_age_s` are NaN when the held sample is stale (older than the max age) or absent, as in tocf.
- `lead_prob` is reported on every fresh radar sample (also when `lead_valid` is False); every other `lead_*` value is NaN
  unless `lead_valid`. `lead_source` is 0 unless `lead_valid`. `lead_a_slope_mps2` uses the radar's vLead on the native samples,
  falling back to (held ego speed + vRel) where vLead is NaN.
- Dead-channel rule: magnitude NaN throughout, or max |magnitude| == 0 over the drive while the matching pressed flag fires on
  >= DEAD_CHANNEL_MIN_PRESSED samples -> column NaN, `*_alive` False (steer torque likewise -> `torque_alive`, raw and norm NaN).
- A held flag is False at a `gap` row when its stream has no sample inside the max age (the causal hold never looks forward), so
  `das_any` can drop for the single gap row; windows containing a gap are ineligible anyway (S6).
- `private/qc/drives_private_qc.parquet` columns = `vtt.drives.PRIVATE_QC_COLUMNS` (real dt median/p01/p99/max, real span,
  gap total, clock drift, raw version string, counts per topic, unknown-union counts, enabled fractions, build time).
- Resume: a drive is skipped when its parquet exists with the row count recorded in the index; rows of drives not rebuilt in a run
  are kept as long as their file exists (so `--routes` subsets and `--limit` never shrink the index); `--no-resume` rebuilds the
  listed routes. Failures go to `work/logs/s05_failures.csv` (route_uid, status, masked error); exit 2 when > 0.5% of the
  attempted routes failed, the index has schema problems / duplicate drive ids, or nothing was built.

## Contract notes

### S3a (vtt/specs.py, vtt/labels.py, vtt/inventory.py, scripts/s03_*.py) — 2026-11-05
- Extra module `vtt/inventory.py` (threshold schemes, platform table, cell tables, markdown rendering) keeps `vtt/labels.py` to the
  per-route label rules. Public names used by other stages: `labels.software_era`, `labels.region_bucket`, `labels.powertrain_map`,
  `labels.size_class / cut_size_classes` (None while `config.FOOTPRINT_THRESHOLDS_M2` is None), `labels.footprint_m2`,
  `labels.rebase(path, data_root)` (re-roots a `vtt.paths.P` path for `--data-root`), `specs.read_platform_specs`,
  `specs.fp2platform_dict`.
- `vtt/specs/fingerprint_to_platform.csv`: `method` manual rows whose `note` does not start with `UNRESOLVED` are hand-resolved
  and are kept verbatim by later `s03_platforms.py` runs; every other row is recomputed. The `suffix` method covers two cases:
  a fork tag from `specs.FORK_SUFFIXES` was stripped, or a trailing model year was matched against the platform's CarDocs years
  ("year-aware", never a blind strip). A spelling equal to a platform name resolves to itself (`fp2platform_dict`).
  The opendbc checkout is parsed with `ast` (brand-specific `*CarSpecs` subclasses, `OTHER.specs` and `OTHER.override(...)` handled);
  with the 2024-08 local fallback 289/484 spellings resolve (64 % of hours); the rest wait for the upstream clone or manual rows.
- `vtt/specs/platform_specs.csv`: one row per platform referenced by the mapping (manual spellings included, so specs can be filled
  for fork names directly). `curb_mass_kg` / `wheelbase_mm` come from CarSpecs (`*_source = opendbc`), `generation_note` holds the
  CarDocs model-year strings, brand/model/body come from the fingerprint-labelled taxonomy routes (empty when unknown, e.g. MOCK).
  Filled cells are never overwritten; rows are never deleted. `footprint_m2` may be left empty: S3 computes it from the three
  dimensions (`footprint_min/max_m2` are informational, used by the inventory only).
- `build/routes_labels.parquet`: `powertrain_raw` is the route-level 7-value code after the ECU rules (Toyota/Lexus hybrid or
  transmission ECU, Honda Accord 2018-22 ECU pattern, VW Golf `direct` transmission), i.e. `MIXED` only when the route could not be
  resolved; `powertrain = config.POWERTRAIN_MAP[powertrain_raw]`. The rule that fired is recorded in `label_note`.
  `label_source` values: `fingerprint | alias | prefix | fw-match | directory` (taxonomy csv), `resolved_new` (no taxonomy row,
  resolved from the probe fingerprint), `unmapped` (fingerprint unknown to the taxonomy tables), `none` (no fingerprint at all);
  `label_grade` A for fingerprint-based sources, B for `directory`, C otherwise. A `mock` fingerprint carries no car information:
  such routes take labels and platform from the folder name (grade B, note "mock fingerprint"). Routes whose fingerprint has no
  mapping row stop the stage with exit 2 (re-run `s03_platforms.py`, which picks up new spellings from `routes_resolved.parquet`).
  `private/labels_private.parquet` is written mode 0600; `country` is the raw route_master value, `in_house` its boolean.
- `build/label_overrides.csv` fields accepted: platform, brand, model, body, size_code, powertrain_raw, powertrain, label_note,
  region, wheelbase_mm, track_front_mm, track_rear_mm, curb_mass_kg, spec_confidence, label_source; derived fields (powertrain,
  footprint, size_class, label_grade) are recomputed after the overrides; an unknown field is an acceptance failure (exit 2).
- `build/SIZE_INVENTORY.md` schemes: `tertiles` = platform-count tertiles of the sorted platform footprints (midpoints between
  ranks N/3 and 2N/3); `natural_breaks` = two largest gaps between consecutive platform footprints weighted by the smaller platform
  count on each side (gap x min(n_below, n_above) / N) with every class holding >= 10 % of the platforms; `--thresholds a,b` adds a
  user scheme; a locked `config.FOOTPRINT_THRESHOLDS_M2` is tabulated as well. Hours = `n_segments / 60` from routes_resolved;
  devices = distinct `dongle_log` of labels_private (counted, never printed).
- All three scripts accept `--specs-dir` (default `vtt/specs`) and `s03_platforms.py` / `s03_labels.py` accept `--taxonomy-dir`
  (folder holding `taxonomy_tables.py`) and `--taxonomy-csv`, so tests run them on synthetic inputs.

## Contract notes
Append-only deviations / additions agreed per module (the stage table above stays authoritative for names).

### S2 (`scripts/s02_extract_all.sh`, `s02c_segments_table.py`, `s02d_accept.py`; modules `vtt/segments.py`, `vtt/extract_accept.py`, `vtt/reuse.py`, `vtt/reroot.py`)
- `s02_extract_all.sh` options: `--data-root DIR`, `--raw-root DIR` (exports `VTT_RAW_ROOT`; tests point it at a synthetic tree), `--dry-run` (prints every
  command, runs nothing), `--status` (runs the extractor's own `--dry-run` per chunk: counts of routes to extract / already done; this is the
  "dry-run reports done 4,843" acceptance line), `--skip-reuse`, `--only-chunk NAME`. Exit code = `s02d_accept.py`'s.
- Reuse step: hardlink probe into `$D/tmp`, then `cp -al -n` (fallback `cp -a -n`) of `config.REUSE_EXISTING_SIGNALS` into `work/signals`;
  skipped when `private/salt.txt` differs from `SEED_PRIVATE_ROOT/salt.txt` (uids would not match). Afterwards `$PY -m vtt.reuse --fix`
  removes reused directories that are not a file-for-file image of their source (the extractor re-extracts them). Marker:
  `work/.reuse_signals_done` (mode, source dir count, date); delete it to redo the reuse step.
- Chunk logs: `build/logs/s02_<chunk>.log` (appended on re-runs), plus `build/logs/s02c_segments.log`, `build/logs/s02d_accept.log`.
- `s02c` writes `work/tables/segments.parquet` exactly as `tocf.schema.SEGMENTS` (rates recomputed with `tocf.decim.measured_rate`) and two
  additional side tables the acceptance and later QC read:
  - `work/tables/segment_counts.parquet`: `route_uid, seg, src_kind, span_s, n_carState, n_radarState, n_pose, n_controlsState, n_carControl,
    n_selfdriveState, n_driverMonitoringState` (rows per topic parquet per segment; `span_s` from the segment's `t0/t1_mono_ns`, NaN when -1).
  - `work/tables/routes_meta.parquet`: `route_uid, version, software_era, device_type, extractor_version, config_version, n_seg, n_seg_qlog,
    n_seg_rlog_dec, n_seg_none, n_truncated, n_read_error, pose_source, car_fp, opLong, radarUnavailable, pcmCruise, altExp` (one row per
    extracted route; internal, never released; `software_era` comes from `vtt.manifest.software_era` when that module exists, else
    `vtt.segments.software_era_fallback` with identical semantics: `0.8|0.9|0.10|0.11` by major.minor, `2025|2026` calendar writers, else `other`).
- `s02d` acceptance definition (constants in `vtt/extract_accept.py`, candidates for `vtt/config.py`): a *full* segment is a qlog segment with
  span >= 55 s (the trailing partial minute and cut files are reported as `n_short`, never judged); a full segment is *good* with
  >= 500 carState and >= 200 radarState rows; per era `good_frac >= 0.99` where `n_full > 0`. Coverage: expected routes = union of
  `build/chunks/routes_chunk_*.csv` (else uid_map); `missing / expected <= 0.005` (missing = expected routes without `route_meta.json`, i.e.
  failed or never extracted). Also rejected: any `extractor_version != config.EXTRACTOR_VERSION`, and stale S2c tables (extracted routes absent
  from `routes_meta.parquet`). Outputs `build/qc/s02_accept.csv` (one row per era + `ALL`, columns in `vtt.extract_accept.ERA_COLUMNS`, with
  the unknown-union histogram bins `unk_0, unk_1_10, unk_11_100, unk_101_300, unk_301p`) and `build/qc/s02_accept.md`.
- `s02b_audit_probe.py`: `private/route_audit.parquet` as contracted (explicit arrow schema `vtt.audit_probe.AUDIT_SCHEMA`; nullable bools).
  `engine_off_moving_frac` = share of samples with vEgo > 5 m/s whose engineRpm == 0 (NaN without moving samples). Resumable: probed uids are
  skipped, `--retry-errors` re-queues rows with a non-empty `err`; `err` is `missing_dir | no_logs | <ExceptionType>[,..] | ''` (a cut file
  is not an error: its readable prefix counts). Exit 2 when > 1 % of the routes probed in a run ended in an error. Caveat for S7: many car
  ports never populate `engineRpm` (a 0.9-era ICE route probed with engine_rpm_pos_frac = 0), so `engineRpm == 0` alone is not BEV evidence.

### S9 (`vtt/package.py`, `vtt/readme.py`, `vtt/gate/{rules,scan,pre_upload,pre_push}.py`, `scripts/s09_package.py`, `scripts/s09b_gate.py`)
- Release tree (`release/<version>`): `README.md`, `LICENSE` (notice + CC BY 4.0 legal code from `docs/LICENSE_CC_BY_4.0.txt`),
  `CHECKSUMS.sha256`, `index/{drives,windows_60s,vehicles,platforms}.parquet`, `index/split_summary.csv`, `index/readme_numbers.json`,
  `splits/*` (copied from `build/splits`), `schema/{drive,drives,windows_60s,vehicles,platforms,windows_shard,official_inputs}.json`,
  `data/vehicle_NNNN/drive_NNNNNN.parquet` (hardlinks to `build/data`, copies on EXDEV; stale files removed),
  `windows/{train,validation,test}/<split>-NNNNN-of-NNNNN.parquet`, `.gate_ok`, `.cache/vtt_checksums.json` (size/mtime digest cache).
- `CHECKSUMS.sha256` = `<sha256>  <relative path>` sorted by path over every file except `.cache/**`, itself and `.gate_ok`.
  `.gate_ok` = `{utc, n_files, checksums_sha256}` written only on 0 findings and deleted on any finding; it is a local marker and
  should not be uploaded (S10 may skip it; the gate allow-list tolerates it either way).
- `index/windows_60s.parquet` is `build/index/windows.parquet` re-written with table name `windows` (every window, incl. ineligible).
  `index/vehicles.parquet` / `index/platforms.parquet` follow `vtt.gate.rules.VEHICLES_COLUMNS` / `PLATFORMS_COLUMNS` (aggregates of the
  drives index; a platform spanning several `split_platform` values is written `mixed`, which the enum gate accepts but the README
  text does not expect). `index/split_summary.csv` is copied from `build/index/split_summary.csv` when S8 wrote one, else computed
  with columns `task, protocol, class, split, windows, hours, groups, platforms`. `index/readme_numbers.json` holds every number
  printed on the card (raw values + row lists; `vtt.readme.fmt` formats them).
- `splits/*.parquet` are re-written with metadata exactly `{vtt_table=<file stem>, vtt_schema_version}` and without pandas index
  columns; `splits.json` is copied verbatim and scanned as text (no paths, dates or ids may appear in it, incl. its `config` block).
  Gate G3 for `splits/*.parquet` and `index/split_summary.csv` = every column name in `vtt.gate.rules.SPLIT_TABLE_ALLOW` and not
  matching `DENY_COL`; S8 must pick names from that set or extend it (append-only).
- Window shards: rows = `eligible & (task_size_eligible | task_bev_ice_eligible) & <split-col> in {train, validation, test}`, ordered by
  (split, drive_id, w_idx), `--split-col split_platform` (default) or `split_device`, `--shard-windows` rows per file (default 5000,
  acceptance error above 500 MB). Columns = `WINDOWS_INDEX_COLUMNS` + one `fixed_size_list<float32>[600]` per `config.OFFICIAL_INPUTS`
  (booleans as 0/1, NaN kept), table name `windows_shard`, zstd, row groups of 1000. A shard whose row count and column names match the
  plan is kept on re-runs (`--force` rebuilds); windows eligible for neither task reach no shard.
- Gate findings are aggregated per (gate, rule, file[, column]) with a hit count; paths / columns / details are masked
  (`vtt.gate.rules.gate_mask`). Hard rules: HEX16, HEX12, ROUTE_DATE, ROUTE_NUM, PATHS, COORD, VIN, MONO, ANON_AUTH, ACRONYM
  (the driver-assistance acronym; only the citation token and the single README bibtex fence are exempt), FORBIDDEN, EXACT (G9);
  allow-listable via `vtt/gate/gate_allow.json`: ISO_DATE per file and the one contact e-mail. Deviation: G3 compares column NAMES
  (exact list) like `vtt.schema.check_public_parquet`; a dtype drift is reported as a warning, not a finding. G5 flags `|x| >= 1e9`
  in numeric and sequence columns outside `COUNTER_ALLOW` and a `t_s` that does not start at 0 or is not strictly increasing.
  Extra tree checks: required files, `index/drives.file` == the `data/` file set, drive-id uniqueness, pseudonym formats.
  Reports go to `private/logs/gate_<utc>.json` (0600) when the private root exists, else `<data-root>/tmp/`; exit 0 iff no findings
  (else 2). The private root is optional: without it FORBIDDEN / EXACT are skipped with a warning.
- Pre-push (`scripts/s09b_gate.py --pre-push <repo>`): the data-directory rule (`private|work|release|.cache`) applies to directory
  components only (`vtt/private.py` is fine); text rules run on tracked files and on every text blob of `git rev-list --all --objects`
  with no allow-list under `tests/`. The current repository text still carries absolute paths (`env.sh`, `vtt/config.py`,
  `docs/PIPELINE.md`) and ISO dates in comments; S11 must parameterize the paths before export (dates are allow-listed per file).
- Template literals allowed to contain digits (`vtt.readme.TEMPLATE_LITERALS`): the license strings, the arXiv id, the bibtex key and
  year, `windows_60s`, `m^2`, `sha256`, `float32`, `0 / 1`, `footprint_m2`, `macro-averaged F1`; everything else is a placeholder.

S4 (`vtt/pseudonyms.py`, `scripts/s04_pseudonyms.py`), appended 2026-11-05:
- `private/pseudonym_state.json` (new private file): per-namespace counter `n_assigned` and `retired` ids plus a fingerprint of
  `release_seed.txt`. Ids are positions of a seeded random permutation of each namespace handed out in sorted key order, so the
  counter is what makes the map append-only; a retired id (a cluster that merged / split under `--allow-regroup`) is never reused.
- `private/inhouse_groups.csv` carries a third column `group_id` (the single lab cluster's pseudonym; empty for a listed device
  without routes). Sources per device are '+'-joined: `route_master`, `tocf_lab`, `tocf_t0_extra`, `misfiled_name_dir`
  (person-name folders of the re-key table; hex-named re-keys are merge evidence only, not lab membership).
- Group evidence = shared VIN hash, shared counter-style route id across devices (date-style ids only within one car dir), explicit
  pairs of `identity_merges.csv`, one lab cluster; a hash / route id shared by more than the extractor's
  `IDENTITY_PLACEHOLDER_MAX_DONGLES` devices is ignored. A re-run that would merge or split existing clusters exits 2 unless
  `--allow-regroup`; a route whose (device, platform) key changed keeps its drive_id and takes the id of its new vehicle key
  (counted as `n_vehicle_rekeyed`). Routes missing from a later `routes_resolved` keep their rows in the private map.
- `build/drive_names.parquet` lists only the routes present in the current `routes_resolved`.

S8 (`vtt/splits.py`, `scripts/s08_splits.py`), appended 2026-11-05:
- `split_platform` has a fourth value `excluded` (drives of a group that also drove a held-out platform, on its non-held-out
  platforms); `split_device` has the three values only.
- Hours = eligible windows / 60; cell = size_class x powertrain with `NA` for an unknown label (20 cells). Group loads per cell are
  (hours, drives, presence) with equal weights in the squared relative-deviation objective.
- H2 cap (`EVAL_GROUP_CAP` x split target) is enforced in every cell whose target is >= 5 h and, in addition, in every supported
  cell (a small supported cell would otherwise be dominated). Supported (device scheme) = >= `min_eval_groups` groups that fit under
  the cap of a target-size split and jointly hold >= `min_eval_hours_frac` of the cell's hours; the hours fraction is scaled by the
  scheme's evaluation share (0.24 for the 30 % device evaluation; 0.12 for the 15 % platform hold-out / the 15 % pool validation).
  After greedy + refinement a repair phase exchanges groups until the asserted band / cap violations of supported cells vanish.
- Platform hold-out: candidates = platforms with >= `PLATFORM_HOLDOUT_MIN_GROUPS` groups, no lab group, not the cell's top platform;
  a subset-sum search over the platforms' in-cell hours (0.5 % resolution, spillover from platforms held out for other cells
  subtracted) picks the subset in the 15 % +- tol band maximising (brand BEV/ICE pairing bonus, closeness, count); when the band is
  unreachable the closest reachable share is taken (possibly nothing) and the cell counts as unsupported for the platform test.
  Validation of the platform scheme = 15 % of the remaining pool's hours (groups without held-out drives), same engine.
- `low_support_cell` (public column) is computed on the headline (platform) test set; the device-scheme flag is in splits.json.
- Extra files next to the contract ones: `eval_balanced_<task>_<split>_device.parquet` and `train_uniform_<task>_device.parquet`
  (device scheme); the unsuffixed files follow the headline platform scheme. Balanced lists: per-class count = min over classes of
  the feasible maximum under the caps; fallbacks `platform_cap_relaxed` then `group_cap_relaxed` are applied only when the count
  would fall below 20 per class and are recorded in `splits.json["balanced_eval"]`; `empty_class` marks a list that could not be
  built.
- `splits.json` extra keys: `headline`, `config_version`, `schema_version`, `low_support_cells`, `class_weights_device`,
  `balanced_eval`, `violations`, `counts`; each cell also carries `split_device`, `split_platform`, `supported_device`,
  `supported_platform_test`, `supported_platform_validation`, `low_support_device`, `shares_device`, `max_group_share_device`,
  `test_share_platform`, `validation_share_of_pool`, `pool_hours`, `holdout` (contract keys `split`, `supported`, `max_group_share`
  refer to the headline scheme). `build/index/split_summary.csv` columns: `scheme, task (all|cell|size|bev_ice), class, split,
  hours, n_windows, n_drives, n_groups, n_vehicles, n_platforms`.
- On a violated assertion nothing public is written; `build/splits/splits_FAILED.json` holds the diagnostics and the script exits 2.

## Contract notes
S4 + S8 fixes after verification (`vtt/pseudonyms.py`, `scripts/s04_pseudonyms.py`, `vtt/splits.py`, `scripts/s08_splits.py`,
`vtt/gate/rules.py`, `vtt/schema.py`, `vtt/package.py`, `docs/README_template.md`), appended 2026-11-05:
- `build/index/split_summary.csv` column `scheme` is renamed `protocol` with the values `split_device` / `split_platform`
  (the split column it describes, as in the S9 fallback summary); columns are now `protocol, task (all|cell|size|bev_ice),
  class, split, hours, n_windows, n_drives, n_groups, n_vehicles, n_platforms`. Before writing, S8 checks every column of
  the summary and of `splits/*.parquet` against `vtt.gate.rules.SPLIT_TABLE_ALLOW` / `DENY_COL` (ValueError, exit 2).
- `splits/*.parquet` lists are written with exactly the release metadata (`vtt_table` = file stem, `vtt_schema_version`),
  no pandas metadata and no index column, so `build/splits` passes the gate as is (S9 may still re-write them).
- `split_platform` value `excluded` is now part of the public contract: `vtt.gate.rules.ENUM_DOMAINS['split_platform']`
  = {train, validation, test, mixed, excluded}, `vtt.schema.DRIVES_INDEX_COLUMNS` describes it, the README template
  explains it, and `index/platforms.parquet` takes a platform's split from its non-excluded drives (`excluded` only when
  every drive of the platform is excluded; `mixed` when its groups span train and validation, which is expected because
  validation of the platform scheme is group-disjoint inside the non-held-out pool, not platform-disjoint).
- Platform hold-out: cells with an unknown size class or powertrain (`NA`) never hold anything out (no task evaluates
  them); labelled cells are swept in decreasing size and the sweep is repeated until no cell changes its choice (at most
  `vtt.splits.HOLDOUT_MAX_PASSES` sweeps), each cell counting the spill-over of every platform held out for the other
  cells; `band_reachable` (and therefore `supported`) is judged on the final share. `splits.json["counts"]` carries
  `n_holdout_passes`, `holdout_converged`; the `holdout` dict of an unlabelled cell carries `skipped`.
- `scripts/s08_splits.py` exits 2 when `private/inhouse_groups.csv` is missing or lists no lab group, unless
  `--no-inhouse` is given (INHOUSE_POLICY is never skipped silently); the printout ends with the lab-group count.
- `scripts/s04_pseudonyms.py` exits 2 when the route_master table is absent unless `--no-route-master`; the printout
  ends with the in-house device count, the explicit merges used / skipped and the vehicle ids retired.
- S4 reads the `evidence` column of `private/identity_merges.csv`: pairs whose evidence is in
  `config.MERGE_SKIP_EVIDENCE` (default `date_id_other_car`, the case S4's own route_id rule refuses) are counted per
  evidence in the printout (`merges.by_evidence`, `n_skipped`) and not merged.
- S4 ignores re-key values that are not 16-hex ids (S1's self-mapped "needs a manual entry" marker), counted as
  `n_misfiled_markers_ignored`, and accepts only 16-hex ids as lab devices (`n_inhouse_nonhex_ignored`), so a folder name
  never reaches `private/inhouse_groups.csv`.
- `private/pseudonym_state.json` now also retires vehicle ids: when every route of a (device, platform) key moved to
  another key (platform re-labelled between runs), the old `vehicle_id` goes to `namespaces.vehicle.retired`
  (`n_vehicle_ids_retired`) and is never handed out again; a later run with unchanged inputs succeeds.

## Contract notes

### S3a fixes after verification (vtt/specs.py, vtt/labels.py, vtt/inventory.py, scripts/s03_*.py) — 2026-11-05
- Per-route label overrides moved to their own file: `build/label_overrides_routes.csv` (`vtt.paths.P.LABEL_OVERRIDES_ROUTES`,
  columns `route_uid, field, value[, reason]`). `build/label_overrides.csv` (`P.LABEL_OVERRIDES`) stays the S7 table
  (`scope, key, powertrain, label_note, resolve`) and is never read by `s03_labels.py`. A file at the S3 path without the
  per-route columns is an acceptance failure (exit 2, counts-only message), never a traceback.
- Override semantics: a `platform` override re-merges that platform's `platform_specs.csv` row (wheelbase, tracks, mass,
  footprint, spec_confidence) before the route's other override rows are applied in file order, so explicit spec cells win over
  the re-merged ones; an explicit `powertrain` override wins regardless of row order, otherwise `powertrain` is recomputed from an
  overridden `powertrain_raw`; footprint, size_class and label_grade are recomputed afterwards.
- The specs data directory is `vtt/specs_data/` (was `vtt/specs/`, which would shadow the module `vtt/specs.py` as soon as it
  became a package): `vtt/specs_data/fingerprint_to_platform.csv`, `vtt/specs_data/platform_specs.csv`; `specs.SPECS_DIR`, the
  scripts' `--specs-dir` default, `.gitignore` and the S9 pre-push CSV exemption (`vtt.gate.pre_push.EXEMPT_CSV_DIR`) follow.
  Read `vtt/specs/*.csv` in the stage table and the file contracts above as `vtt/specs_data/*.csv`. The research sheet
  `vtt/specs_data/specs_research.csv` (spec-research workflow output, free text with URLs) is git-ignored: it is a working file
  for filling `platform_specs.csv`, not a release table.
- `fingerprint_to_platform.csv` also carries one row per distinct car-model folder spelling (`car_dir`) of the taxonomy's
  `directory` / `fw-match` rows (Unknown and MOCK excluded), because the labels stage falls back to the folder spelling for
  routes without a usable fingerprint. `s03_platforms.py` fails (exit 2) when a taxonomy fingerprint or folder spelling has no
  row; `s03_labels.py` fails the same way for a folder spelling without a row (same remedy: re-run `s03_platforms.py`) and prints
  the count of affected routes. Real data: 110 folder spellings, 19 of them not seen as a fingerprint (9 resolve by exact /
  migration / suffix, 10 are manual rows); the mapping now has 503 rows, the specs table 364 platforms (+8).
- Routes without a usable fingerprint (`directory`, `fw-match`): platform from the folder spelling, else the unique
  `platform_specs.csv` platform carrying the taxonomy model (`labels.platform_from_model`; when several platforms share the model
  the ones whose own powertrain contradicts the route's `powertrain_raw` are dropped first); `fw-match` rows (folder Unknown, mock
  fingerprint, model from the firmware match) try the model first. The chosen rule is recorded in `label_note`
  ("platform taken from the folder name" / "... from the taxonomy model, its only platform"); an ambiguous model leaves the
  platform null with the note "platform ambiguous: N platforms share the model". `fw-match` keeps grade A (the model and
  powertrain come from the route's own firmware), so an A-grade route may have a null platform only in that ambiguous case.
- `inventory.natural_breaks` scores every admissible pair of gaps (the earlier shortcut assumed the optimum contains a top-scoring
  gap, which the min-share constraint can break); ties go to the lower thresholds. Verified against a brute-force reference.

### S6 (`vtt/windows.py`, `scripts/s06_windows.py`, `vtt/features_core.py`) — 2026-11-05, revised after verification
(The stage builder had inserted this note into the S1 block above; it was moved here, append-only from now on, and the text below
supersedes the earlier version.)
- Windows are consecutive non-overlapping blocks of `WINDOW_SAMPLES` rows from row 0 of each drive; the trailing partial block is
  listed too (`eligible=False`, `inelig_reason='short'`) so every sample belongs to exactly one window and drives shorter than 60 s
  still appear (`n_windows` per drive = ceil(n_rows / 600)).
- `span_s` is the nominal span `(rows - 1) x 0.1 s` (59.9 s for a full window) and the schema description says so: public tables
  carry nominal time only, so the 60-s guarantee is `gap_count == 0` (no `gap` row strictly inside the window; a `gap` flag on the
  window's first row marks missing data *before* the window and does not count). `config.WINDOW_SPAN_S` is therefore not tested by
  S6 and there is no `span` reason; `inelig_reason` joins every failed test in the order `short+gap+stationary` and is null for
  eligible windows. The real per-drive span stays in S5's private QC frame.
- `kind`: human (`das_any_frac == 0`), das (`== 1`), mixed. `human_only_strict`: `human_only` and no `das_any` sample in the
  `HUMAN_GUARD_S` (100 rows) before the window; vacuously true for the first window of a drive (the ignition cycle starts there).
- `human_only_windows.parquet` = rows of `windows.parquet` with `eligible AND human_only_strict` (same columns, same table name
  `windows`).
- `n_accel_ep` / `n_decel_ep`: `vtt.features_core.accel_episodes` / `decel_episodes` on the window's own `v_mps` (EV-paper rule at
  10 Hz: Savitzky-Golay `sg_win=5, sg_order=2` least-squares kernel with polynomial edge extrapolation; maximal monotone runs
  tolerating `eps_mps` dips; `>= min_s / dt` samples; `|dv| >= min_dv_mps`; mean acceleration `|dv| / duration >= min_mean_a`
  computed from the smoothed speed, i.e. the kinematic mean of `a_mps2`, robust to vehicles with a degenerate aEgo channel).
  Runs are clipped at the window bounds (the window is self-contained; the starter kit reproduces the count from the 600 steps).
  Known bias of the literal rule (stated in the schema description): a monotone run absorbs the steady cruise around a manoeuvre,
  so a short acceleration inside a long steady cruise (e.g. +2.5 m/s in 3 s within a 60-s highway cruise) is tested with the cruise
  in its duration and is not counted, while the same manoeuvre in stop-and-go traffic is. `config.ACCEL_EP['trim_win_s']` (default
  0 = literal rule) is the switch for the parent/user: with e.g. 2.0 s the ends of each run whose central slope of the smoothed
  speed over that window is below `min_mean_a` are trimmed before the tests (`features_core.active_slope_mask`), so the count depends
  on the manoeuvre only; the starter kit's fallback implements the literal rule, so flipping the switch requires porting the trim
  there too (the starter kit delegates to `vtt.features_core` whenever it is importable).
- `n_lead_accel_events`: `vtt.features_core.lead_accel_events` inside the window: lead episodes = runs of `lead_valid`; onset when
  `|lead_a_slope_mps2| < quiet_abs` on every sample of the `quiet_s` before it and `>= slope_on` on every sample of the `hold_s`
  after it (NaN fails both); the 8-s response horizon is *not* required for the count (`horizon_s=None`); the starter kit's
  `ego_response_delay` uses `resp_a / resp_hold_s / horizon_s`.
- `cf_frac` = share of valid-lead samples with `thw_s <= THW_CF_S`; `thw_med`, `lead_speed_mean` over valid-lead samples;
  `yaw_abs_mean` over `yaw_valid` samples; all NaN when the denominator is empty.
- Resume cache: per-vehicle statistics parts under `work/windows_parts/<vehicle_id>.parquet` (never released). A part is reused
  only when it is newer than every drive file of the vehicle AND holds rows for every drive the index currently assigns to that
  vehicle (`vtt.windows.part_is_fresh(part, paths, drive_ids)`; a superset is fine), so a part written by a `--limit` / `--routes`
  smoke run is rebuilt by the next full run. At assembly `vtt.windows.index_rows` keeps only rows whose `(drive_id, vehicle_id)` pair
  is in the drives index (count printed as `stale_rows_dropped`), so drives removed from the index (exclusions before S8, subsets,
  pseudonym changes, S4 vehicle re-keys) never reach the window tables. `acceptance_problems` additionally fails on windows of
  drives not in the index, vehicle_id mismatches and null `group_id` where the index has one; the 'drives with rows but no windows'
  message names the `--force` remedy. Labels / `group_id` / split columns are joined from `drives.parquet` at assembly, so S7 / S8
  label changes only require re-running the (cheap) assembly.
- The drive file is resolved from the index `file` column relative to `build/` first, then `build/data/`, else
  `data/<vehicle_id>/<drive_id>.parquet`.
- `--data-root` overrides are implemented with `vtt.windows.Roots` (a rebased mirror of `vtt.paths.P` for build/, work/, private/).
- Tests (`tests/windows_tests.py`, `tests/qc_tests.py`, `tests/features_core_tests.py`) remove their temp trees under `$D/tmp`.

### S7 (`vtt/qc.py`, `scripts/s07_qc.py`) — 2026-11-05, revised after verification
- carState rate: from `private/qc/drives_private_qc.parquet` when it has `drive_id` and one of
  `cs_rate_hz | rate_carState | ...` (rate), `cs_dt_med_s | dt_med_s | ...` (median dt) or `real_span_s | ...` (real span);
  flags `rate_low` (< 8 Hz) / `rate_high` (> 12 Hz). Without it: `rows_per_seg = n_rows / (n_segments - 1)` for drives with
  >= 3 segments, flag `rows_per_seg_low` (< 480).
- Lead flags: `lead_radar_never` (radar available but never the source over >= 1000 valid-lead samples), `no_lead` (no valid lead
  over > 10 min moving). The earlier `lead_source_mismatch` flag was dropped: S5 derives `lead_source` through
  `tocf.lead.lead_source` with the route's `radarUnavailable`, so a radar-sourced sample on a `radar_unavailable` drive cannot exist
  and the flag was vacuous. The contract item 'lead_source distribution vs radar_unavailable' is met by `vtt.qc.lead_source_table`
  (per `radar_unavailable` value: drives, drives with a lead, valid-lead samples, sample-weighted radar / camera shares), rendered
  in REVIEW.md §5 and printed by the script. `yaw_all_zero`: every finite yaw sample is 0 (a boolean column in `drives_qc.csv`;
  a drive without a scan row counts as False). Duplicates: same `n_rows` and sha1 of the `v_mps` float32 bytes -> `duplicate_of` =
  first drive_id of the group (the sha itself stays in `work/qc_parts`). Multi-platform devices use the private pseudonym map
  (`dongle_log`) when present, else `group_id`; only counts are printed.
- Label audit verdicts (`label_audit=<verdict>` token appended to `label_note`, idempotent): `ok`, `conflict_bev_engine_rpm`
  (BEV, `engine_rpm_pos_frac > 0.05`; rpm == 0 is the consistent BEV reading -> `ok`), `conflict_ice_electric` (ICE,
  `regen_any or charging_any`, independent of the rpm channel), `no_rpm_signal` (ICE / HEV / PHEV with `engine_rpm_pos_frac == 0`:
  many car ports never fill `carState.engineRpm`, so rpm == 0 while moving is an unpopulated channel, not an engine-off fact; see the
  S2b caveat), `suspect_mild_hybrid` (ICE, rpm > 0 and `engine_off_moving_frac > 0.3`), `suspect_hybrid_engine_always_on` (HEV/PHEV,
  rpm > 0 and `engine_off_moving_frac <= 0.001`), `no_audit` (no probe row or no `drive_names.parquet`), `no_audit_signal` (probe
  `err` or NaN rpm), `na` (null powertrain). Only `conflict_*` verdicts block (exit 2) unless resolved by an override or
  `--allow-conflicts`; `suspect_*` are warnings; `no_*` verdicts are informational (not flagged, not in the suspect list).
- `build/label_overrides.csv` (optional): columns `scope,key,powertrain,label_note,resolve`; `scope` in drive | vehicle | platform,
  `key` the matching pseudonym / platform; `powertrain` blank = keep, ICE/HEV/PHEV/BEV = relabel (also recomputes
  `task_bev_ice_eligible`; `powertrain_raw` is untouched), `null` = clear; `label_note` text is appended as `override:<text>`
  (masked; identifier-like tokens are rejected); `resolve=1` marks the matched conflicts `.../resolved`. Changed powertrains are
  propagated into `windows.parquet` and `human_only_windows.parquet` when they exist.
- Outputs: `build/qc/drives_qc.csv` (one row per drive: pseudonyms, platform, counts, `cs_rate_hz`, `rows_per_seg`, `rate_flag`,
  alive flags, `yaw_all_zero`, lead shares and `lead_flag`, `duplicate_of`, `multi_platform`, `n_windows`, `n_eligible_windows`,
  `label_audit`, `label_resolved`, `flags`), `build/qc/REVIEW.md` (cells x drives / vehicles / groups / hours / top-platform share;
  hours by cell x region; DAS share and eligible-window hours per cell; dead channels overall and per platform; signal flags and the
  lead-source table; label audit with conflicts by platform and the flagged drive list), `private/qc/label_audit_private.csv`
  (route_uid + probe numbers, mode 0600). `drives.parquet` is rewritten with the same table name it carried.
- Per-vehicle scan parts are cached under `work/qc_parts/<vehicle_id>.parquet` (columns `vtt.qc.SCAN_COLUMNS`, incl. `vehicle_id`)
  with the S6 freshness and ghost-row rules (`part_is_fresh` with the vehicle's drive ids, `index_rows` at assembly). `--dry-run`
  writes nothing at all: stale parts are scanned in memory and no output file is touched.

### S9 fixes after verification (`vtt/package.py`, `vtt/gate/{scan,rules,pre_push}.py`, `scripts/s09_package.py`, `docs/README_template.md`) — 2026-11-05
- Shard reuse is content-based, not count-based: `release/<version>/.cache/vtt_shards.json` maps each shard path to
  `[fingerprint, size, mtime_ns]`, where `vtt.package.shard_fingerprint` = sha256 over the shard schema names, the sequence
  length, every `WINDOWS_INDEX_COLUMNS` value of the planned rows (labels, splits, membership, order) and the size + mtime of each
  drive table the shard reads. A shard is kept only when the entry matches and the file on disk is the one that was written;
  a relabel, a split change or a rebuilt drive table rebuilds exactly the affected shards without `--force` (test:
  `test_relabel_without_force_rebuilds_only_the_affected_shards`). The cache is updated after every shard (resumable) and
  never leaves `.cache/` (skipped by the gate and by CHECKSUMS). `release/data` copies (EXDEV or `--copy`) are compared by size and
  nanosecond mtime, hardlinks by inode.
- Gate parquet scan (`vtt.gate.scan.check_parquet`): files are read in row batches of about `SCAN_BATCH_BYTES` (32 MiB; never a
  whole table), so the peak memory of a 500 MB shard scan is one batch per worker. Numeric magnitude, numeric enum, sequence
  length / magnitude and the `t_s` start / strict monotonicity are checked over every row of every file; the distinct string values
  of the index tables (`index/*.parquet`) and of the splits tables are checked in full; only the drive tables and the window shards
  (`SAMPLED_TABLES`) take the `--sample-rows` bound (default 50,000 rows) for their string columns.
- Private token set (`vtt.gate.rules.private_tokens`): `rel_path` / `copies_rel` contribute the whole path string plus their dongle-folder
  and route-id components, never the first component (the raw car-model folder is a public platform spelling: 91 of 110 folder
  spellings equal a fingerprint string and 320 of 503 mapping spellings equal their canonical platform name, so adding it would
  block the release on G9/EXACT). Dongle folder names (possible person names), misfiled folder names and every id column are kept.
- `scripts/s09_package.py`: every error while assembling (missing drive table, template / numbers mismatch, unreadable parquet) is
  one masked line `s09: FAILED (<Type>): ...` and exit 2; `--debug` re-raises with the traceback.
- Pre-push policy for the code repository (`vtt.gate.pre_push.NAME_ONLY_RULES = ('COORD',)`): the coordinate / origin word rule
  applies to names only (path components, JSON keys, CSV headers, column names of tracked parquet files such as `sample/**`), not
  to the free text of source files and docs, because the pipeline code must name the private columns it reads (`country`,
  `in_house`, `lat` / `lon` of the route master) and the docs must describe them; a coordinate value cannot be caught by a word rule
  anyway. MONO was already a name-only rule. All identifier rules (HEX16, HEX12, ROUTE_DATE, ROUTE_NUM, PATHS, VIN, ANON_AUTH,
  ACRONYM, FORBIDDEN, EXACT) and the soft rules (ISO_DATE, EMAIL) stay on all text, `tests/` included, in tracked files and in every
  history blob. Tracked parquet files additionally get their column names checked (MONO / COORD / VIN + text rules). The release
  gate (`vtt.gate.scan`) is unchanged: COORD applies to all its text, string values and column names. Consequence for S11: the
  PATHS hits of the raw repository are removed by its path rewrite; the remaining literal hits in other modules' tests must be built
  at run time by their owners (a 12-hex uid literal in `tests/s03_scripts_tests.py`, an ISO date in `tests/inventory_tests.py`, the
  bare driver-assistance acronym in `tests/qc_tests.py`), since nothing under `tests/` is allow-listed.
- `docs/README_template.md`: the citation paragraph no longer names the other corpus or uses the citation token in prose; the token
  appears only inside the single bibtex fence (test: `test_citation_token_appears_only_inside_the_single_bibtex_fence`).

## Contract notes
(s10_publish_kit: S10 upload, S11 GitHub export, S12 starter kit + audits)
- `scripts/s10_upload.py readme-validate` runs local card checks (front matter, `license: cc-by-4.0`, a `windows` config, the
  acronym rule) and then `DatasetCard.validate`, which POSTs the card text to the Hub's validation endpoint (not a purely local
  call); `--offline` skips it. `check` compares the repository file list with the local release set plus `.gitattributes`, sizes
  from `dataset_info(files_metadata=True)`, sha256 of README, CHECKSUMS, every `index/` and `splits/` file and `--n-random` (8)
  files from `data/` and `windows/` downloaded into `$HF_HOME/check`; it asserts gated == manual, private == False, card license
  == cc-by-4.0 and streams one row of config `windows`, split `validation`, when `datasets` is importable. No subcommand changes
  repository settings.
- `scripts/s11_github_export.py` rewrites local absolute paths to placeholders in every text file of the export, including the
  defaults of `vtt/config.py` (`$VTT_DATA_ROOT`, `$VTT_RAW_ROOT`, `$VTT_TOCF_ROOT`, ..., `$R`, `$PY`, `$CPY`, `$HOME`), because the
  code gate's path rule has no exemption for `config.py`; `--keep-config-paths` restores the original text (the gate then fails).
  `sample/` comes from `vtt.synth.write_sample(out_dir, n_vehicles, n_drives)` when that module exists, else from
  `vtt.synth_sample.write_sample` (same signature); cap 500,000 bytes. Before `git init` a pre-commit scan runs
  `vtt.gate.rules.scan_text` on every text file (tests/ without allow-list) plus a leftover-path rule; any hit means exit 2 and
  nothing is committed. After the commit `vtt.gate.pre_push.run(out, private_root, python)` decides; a missing gate module is a
  loud failure. The script prints the `ssh -T`, `git remote add` and `git push` commands and never pushes.
- Evaluation protocols are specified in `docs/PROTOCOLS.md` and implemented in `starter_kit/eval.py`: `test_full` =
  `eligible & <task eligibility> & (split_col == "test")` with a class label of the task, `test_balanced` =
  `window_id.isin(splits/eval_balanced_<task>_test.parquet.window_id)`, `test_human_only` = `test_full & human_only_strict`;
  headline metric = macro-F1 outside low-support cells (`macro_f1_supported`). The README template's Protocols table must stay in
  agreement with that file.
- `starter_kit/features.py` delegates acceleration / deceleration episodes and the ego response delay to `vtt.features_core` (so
  counts agree with the windows index) but detects lead-acceleration onsets locally with a 0.5 s transition allowance between the
  quiet window and the hold (`lead_accel_onsets(..., trans_s=0.5)`): with a 1 s least-squares lead slope the strict rule of
  `vtt.features_core.lead_accel_events` (quiet window immediately adjacent to the hold) misses most onsets (3 vs 19 on three real
  routes), so `n_lead_accel_events` in the windows index and `n_catchup_events` of the starter kit are not the same count.
- `scripts/s12_audits.py` reads `build/index/{drives,windows}.parquet` and writes `build/qc/audit_nuisance.csv`,
  `build/qc/audit_inputs_brand.csv`, `build/qc/audit_numbers.json` (keys `nuisance_all_<target>_macro_f1 / _chance_f1`,
  `inputs_<target>_macro_f1 / _chance_f1 / _n_classes`) and `build/qc/AUDITS.md`; the README renderer may pull the limitation
  bullets / numbers from there. Split targets are skipped while the split columns are null (before S8).

### S2 fixes after verification (`scripts/s02_extract_all.sh`, `vtt/reuse.py`, `vtt/segments.py`, `vtt/extract_accept.py`, `scripts/s02d_accept.py`, `vtt/audit_probe.py`, `vtt/config.py`) — 2026-11-05
- `--status` is read-only: it reports the reuse state (the marker text, or "pending" plus a read-only `vtt.reuse` count when
  work/signals exists) and runs the extractor's own `--dry-run` per chunk; it never links, copies, removes or writes the marker,
  and it skips the S1 gate. The pre-flight checks (uid_map, salt, chunk dir) apply to `--status` as well.
- Pre-flight (real mode) reads `build/logs/s01_summary.json`: `accepted` not true -> exit 1 before anything is touched;
  `--force` continues; a missing summary is reported, not fatal (synthetic / ad-hoc roots).
- Reuse step: `mkdir -p work/signals` first; the hardlink probe links a source `route_meta.json` into `work/signals` itself
  (deviation from the plan's `$D/tmp`: on the merged file system only the real destination is representative); the bulk
  `cp -al -n` (or `cp -a -n` after a failed probe) runs only while work/signals holds no route directory; then
  `$PY -m vtt.reuse --src .. --dst .. --fix --fill` compares names + sizes, removes incomplete reused dirs, keeps "fresh" dirs
  (route_meta.json newer than the source's = this build's own extraction, never compared or removed) and rebuilds every source
  route still missing one directory at a time (os.link per file, shutil.copy2 fallback, `.<uid>.reusing` temp dir renamed into
  place). The marker `work/.reuse_signals_done` is written only when `vtt.reuse` exits 0 (`missing=0`, i.e. every source route
  with a route_meta.json is complete or fresh in work/signals); otherwise a WARNING is printed, the next run retries (fill only,
  no bulk cp) and this run's extractor rebuilds the missing routes from the raw library.
- `vtt.reuse` result keys: `source_dirs, source_usable, dest_dirs, complete, fresh, incomplete, removed, filled,
  filled_linked_files, filled_copied_files, fill_failed, missing, hardlinked`; exit 0 iff `missing == 0`.
- `finish()`: a non-zero `s02c` exit (unreadable routes or a crash) makes the script exit 2 whatever `s02d` says;
  `build/qc/s02_accept.csv` is written atomically like the .md.
- `vtt.segments.software_era` normalises the owner's answer to `config.SOFTWARE_ERAS` (`2027.x` -> `other`, identical to
  `software_era_fallback`); `vtt.extract_accept.era_table` judges every era present in routes_meta (`eras_present`: tuple order
  first, then any other value sorted) instead of the fixed tuple only, so no segment escapes the per-era rule.
- `vtt/config.py`: `REUSE_EXISTING_SIGNALS` / `SEED_PRIVATE_ROOT` accept the env overrides `VTT_REUSE_EXISTING_SIGNALS` /
  `VTT_SEED_PRIVATE_ROOT` (the tests point them at a synthetic extraction to exercise the reuse path); defaults unchanged.
- `s02b_audit_probe.py` assumes one writer per route (carState layout from the route's first carState message, carParams from
  the first segment holding one); a field that appears only in a later segment would stay NaN.
- Observation from the 20-route real check (counts only): one 2026-era route with `radarUnavailable=False`, `opLong=True` logs
  radarState at ~1.9 Hz (109-118 rows per full segment), so its 4 full segments fail the `>= 200 radarState rows` rule and the
  2026 era of that sample falls to 0.947. If the full build rejects on this, the rule (not the extraction) needs a ruling: a
  per-route radar-rate exemption (`rate_radarState` < 2.5 Hz for the whole route) or a lower `MIN_RADAR_ROWS` for 2026 writers.

### s10_publish_kit fixes after verification (`scripts/s11_github_export.py`, `starter_kit/{load,eval}.py`, `vtt/synth_sample.py`, `vtt/audits/*`, `vtt/config.py`, `vtt/gate/rules.py`, `.gitignore`) — 2026-11-05
- S11 selects files the way `git add -A` would: the repository's `.gitignore` is evaluated by `git check-ignore` (throw-away
  repository holding only the ignore file), so ignored paths are neither copied nor scanned (the printed count says how many).
  After the commit `git ls-files` must equal the export directory (marker excluded); otherwise exit 2 (`tracked_mismatch`). The
  marker `.vtt_export.json` is written before anything else (`status`: started / refused / committed / gate_failed / ok), so a
  refused export is replaced by the next run without `--force`; refusals return exit 2 (no `SystemExit`). Text rewrites keep
  file modes (`shutil.copymode`). Default author name = the owner of `config.GITHUB_REPO`; `--author-email` help documents the
  `<id>+<user>@users.noreply.github.com` form.
- `.gitignore`: `!sample/**` moved after the `data/` ... `github_export/` directory rules (git applies the last matching pattern,
  so the earlier order dropped `sample/data/**` from every commit), and only the two public specification tables
  (`vtt/specs_data/{fingerprint_to_platform,platform_specs}.csv`) are re-included; every other CSV in that folder (research /
  fill sheets with raw identifiers) stays untracked. A new public table there must be re-included explicitly.
- `vtt/gate/rules.py` PATHS: the four absolute prefixes of `PATH_PREFIXES` now count only when not preceded by a word
  character, `$`, `}`, `>` or `.` (relative fragments such as `build/<data>/...`, `<data-root>/tmp/`, `$D/tmp/x`, `./data/` are
  not leaks). Same rule in both gates; name and counts of the rule unchanged otherwise. Three test files were de-literalised
  for the pre-push gate (tests/ has no allow-list): an ISO date (`tests/inventory_tests.py`), the acronym (`tests/qc_tests.py`)
  and a 12-zero uid (`tests/s03_scripts_tests.py`); behaviour unchanged.
- Balanced lists per split scheme: `starter_kit.load.balanced_list_path / load_balanced_ids(..., split_col)` read
  `splits/eval_balanced_<task>_<split>.parquet` for `split_platform` and `..._device.parquet` for `split_device` (S8 naming);
  `iter_windows` passes `split_col` through. `starter_kit.eval.protocol_mask(test_balanced)` = `test_full & isin(list)` with a
  warning when ids fall outside `test_full` (list of the other scheme). `vtt.synth_sample.write_sample` writes both lists.
- `vtt/config.py`: `OPENDBC_LOCAL_FALLBACK`, `VENV_PY`, `CONDA_PY` take the env overrides `VTT_OPENDBC_LOCAL`, `VTT_VENV_PY`,
  `VTT_CONDA_PY` (defaults unchanged); README.md lists every placeholder the export can emit.
- Audits: `vtt.audits.common.macro_f1` is integer-coded (bincount; 200k windows x 100 classes in 0.12 s instead of 2.2 s, same
  values), `to_str` is vectorised, `chance_levels` permutes codes (identical values for the same seed) and is computed once per
  target and shared by every predictor / method row (`_row(..., ch)`, `_result(..., ch)`).

## Contract notes

### Hold-out unit, spec fill and extraction acceptance (6 Nov 2026, integrator)
- User ruling of 5 Nov 2026 (evening): routes with an unknown car model are not sampled; labels of known models follow the lab
  taxonomy as it stands. Consequently `config.HOLDOUT_KEY = "model"`: `split_platform` holds out whole taxonomy car models
  (`vtt.splits.build_population` renames that column to its internal `platform` name; every `platform` value in
  `splits/*.parquet`, `split_summary.csv` and `splits.json` is therefore a car model). The public `platform` column of the
  index tables keeps the fingerprint-platform name.
- `scripts/s03b_specs_fill.py` fills empty cells of `vtt/specs_data/platform_specs.csv` from the research tables
  (`specs_research_merged.csv`, `specs_fill_pass4.csv`): spelling-level rows first, then car-model rows, then tracks
  estimated as 0.86 x overall width (`track_source = est_0.86xwidth`, confidence C). Filled cells are never overwritten.
- S2 acceptance (`vtt/extract_accept.py`): a full segment is good with >= 500 carState rows and >= 60 radarState rows
  (radar present); fewer than 150 radar rows is reported as `n_low_radar` only (several device generations log radarState at
  2-3 Hz). Per-era good share >= 0.98; the catch-all era `other` is reported, not judged.


## Additions after the first full build (6 Nov 2026)

- **S3c** `scripts/s03c_lock_thresholds.py` writes `docs/SIZE_CLASSES.md` (definition, rationale from `docs/size_classes_rationale.md`,
  cell table, platforms near each threshold, models whose generations fall in two classes, models per class) and, with
  `--write-config`, sets `config.FOOTPRINT_THRESHOLDS_M2` and `config.SIZE_CLASSES_SHA256`. Locked values: 4.43 / 4.88 m^2.
- **Model merges** `vtt/specs_data/model_merges.csv` (from_model, to_model, reason) is applied by `s03_labels.py`
  (`labels.apply_model_merges`): the same platform family recorded under two taxonomy names would otherwise be held out
  separately and exclude every device that drove both (Rivian R1S -> Rivian R1T/R1S). Only `model` changes; `label_note` records it.
- **S5 `--relabel`** (`drives.relabel_index`): refreshes the label-derived index columns from `build/routes_labels.parquet`
  without reading a drive table; split columns are reset, so S6 -> S7b -> S7 -> S8 follow.
- **S9** drops drives with an unknown car model (`config.DROP_UNKNOWN_MODEL`, user ruling) from the release.
- **S13** `scripts/s13_export_tensors.py` writes the modelling cache `work/tensors/X_f16.npy` ((N, 600, 20) float16 of the
  official channels, NaN kept) + `windows.parquet` (window_id order) + `channels.json`; `experiments/` holds the
  classifiers (`data.py`, `models.py`, `train.py`, `run_all.py`, `features_baseline.py`) and `experiments/RESULTS.md`.
  The cache is never released; labels and splits are joined at training time by window_id.
