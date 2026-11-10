# commaCarSegments pipeline (selection, decoding, windows)

Status 10 Nov 2026 (decode and scoring of the budget list done; see "Decode result" below). Three scripts in `experiments/transfer/`: `ccs_platforms.py` (key folding + labels, shared),
`ccs_select.py` (the owner's download list, report `CCS_SELECTION.md`) and `ccs_decode.py` (rlog.zst -> 20 channels ->
windows through `common.py`). Verified on the five sample segments of the verification pass (Ioniq 5, Tesla Model 3
new port, Rivian R1, RAV4 TSS2, Bolt EUV); the bulk download was the owner's step (`CCS_SELECTION.md`, or
`download_owner_sets.sh --only ccs`, which calls `ccs_select.py` itself when `select.txt` is missing); the budget list
(`select_budget.txt`, 49,563 segments) was downloaded, decoded and scored on 10 Nov (results in `RESULTS.md` 2.12).

## What runs where

| step | interpreter | inputs | outputs |
|---|---|---|---|
| `ccs_select.py` | `$CPY` (miniconda; pandas only) | `$EXT/ccs/database.json` (index, 9.4 MB), `$EXT/ccs/shared_dongles.txt` (464 lab-shared devices), `build/index/drives.parquet` (model strata, powertrain per platform), `vtt/specs_data/platform_specs.csv`, `specs_external.csv`, the taxonomy DIRS table; opendbc MIGRATION and platform names are parsed from source with `vtt.specs` (no import) | `$EXT/ccs/select.txt`, `select_budget.txt`, `select_summary.json` (data disk, carry device ids) and `CCS_SELECTION.md` (repo, counts only) |
| `ccs_decode.py` | `<data-root>/ext/venv_opendbc/bin/python3` (Python 3.12.4: opendbc 0.3.1 from PyPI, pycapnp 2.1.0, zstandard 0.25, numpy, pandas, pyarrow) | `$EXT/ccs/segments/<dongle>/<route>/<seg>/rlog.zst`, the cereal `log.capnp` of the openpilot checkout (`$VTT_CEREAL_DIR`, default: the `cereal` folder of the openpilot checkout beside the repository, `<repo parent>/openpilot/cereal`), the repo's `common.py` + `ccs_platforms.py` | `$EXT/<dataset>/work/windows.npz` + `meta.parquet` + `windows_summary.json` (contract), `segments_qc.csv`, `decode_summary.json` |
| `score.py` | `$CPY` (torch, GPU) | the saved windows | `score__*.md/json`, `probs__*.npz` |

Commands (`source env.sh` first; `$EXT` = `<data-root>/external`):

```bash
# 1. selection (already run; re-run after a new index)
nice -n 10 $CPY experiments/transfer/ccs_select.py --index $EXT/ccs/database.json --out $EXT/ccs/select.txt
# 2. owner download: see CCS_SELECTION.md (or: bash experiments/transfer/download_owner_sets.sh --only ccs)
# 3. decode everything that is on disk (routes are the unit of work; 6 workers ~ 5-6 h for the full list)
nice -n 10 $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_decode.py \
    --segments-root $EXT/ccs/segments --list $EXT/ccs/select.txt --dataset ccs --workers 6
# 3b. the five verification segments (done; the result is the scorer's consistency target)
$D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_decode.py \
    --segments-root $D/scratch/ccs_verify/dl/smp --dataset ccs_sample --copy-sample-to $EXT/ccs/work
# 4. score
$CPY experiments/transfer/score.py --dataset ccs --task both
```

## The opendbc version problem and the chosen route

* The lab's openpilot checkout (Aug 2024, `$PY`) decodes the logs with its C++ CAN parser (1.8 s CPU per segment) but
  its opendbc knows 207 platforms: 18 folded index keys are missing, among them the 2025 Tesla port
  (`TESLA_MODEL_3`, `TESLA_MODEL_Y`), `RIVIAN_R1_GEN1`, `HYUNDAI_KONA_2022`, `CADILLAC_XT4`, `GMC_YUKON`, `TOYOTA_YARIS`,
  `TOYOTA_RAV4_PRIME`, `TOYOTA_SIENNA_4TH_GEN`, `CHEVROLET_TRAVERSE`, `HONDA_PILOT_4G`, `FORD_ESCAPE_MK4_5`. Its MIGRATION
  table also lacks the Rivian fold.
* Chosen: a separate venv with the current PyPI opendbc (`python3 -m venv $D/ext/venv_opendbc && pip install
  opendbc zstandard pycapnp numpy pandas pyarrow`). opendbc 0.3.1 has 245 platforms, a 279-entry MIGRATION (adds
  `RIVIAN_R1_GEN1/GEN2 -> RIVIAN_R1` and `VOLKSWAGEN JETTA 6TH GEN`), a pure-Python CAN parser (DBC files generated at
  import), `CarInterface(CP)` / `RadarInterface(CP)` taking `list[(logMonoTime, [CanData])]`. After MIGRATION only
  `HONDA_CRV_HYBRID_6G` (3 segments) and `HYUNDAI_SONATA_HEV_2024` (1) have no interface and are skipped at decode time.
* The openpilot `LogReader` does not import in that venv (it pulls the openpilot runtime: `zmq`, then `smbus2`, and
  the checkout's own opendbc wants `usb1`); the decoder therefore reads the logs with 15 lines of its own:
  zstandard decompress -> `log.Event.read_multiple_bytes` on the cereal schema loaded through a **private**
  `capnp.SchemaParser` (cereal's `car.capnp` and opendbc's `car.capnp` carry the same Cap'n Proto ids, so loading both
  in pycapnp's global parser aborts the process). `carParams` is re-read with opendbc's `CarParams`
  (`as_builder().to_bytes()` -> `from_bytes`, a wire-level copy that keeps every field), its fingerprint folded with
  MIGRATION, then the brand's `CarInterface` and `RadarInterface` are built from it (2025 flags such as Rivian
  GEN1/GEN2 come from the logging version's carParams, as they should).
* `common.py` (and `vtt.config`) import under that venv; cold imports from the data disk are slow under load (the
  machine ran at load average ~35 during these tests: 60 s for `opendbc.car.structs`, up to 270 s for pandas / pyarrow
  on first use), warm imports are seconds. The first segment of a worker pays the brand-module import and DBC
  generation (14 s on the Ioniq 5 sample), later segments do not.

## Channel mapping (per segment, then per route)

Grid: 600 samples from the first CAN packet, step 0.1 s; a sample takes the latest CarState (ZOH, max age 0.3 s = the
lab gap rule); a segment with any stale sample is dropped (none of the five). Routes = all segments of a route present
in the run, concatenated with `t = seg x 60 + k / 10`, cut by `common.cut_windows(align="drive")`: one window per
complete segment, `w_idx` = segment number, `t0_s` = segment x 60; the moving rule (>= 20 % of samples with v >= 1 m/s)
is applied there.

| channel(s) | source | status / note written to `meta.note` |
|---|---|---|
| `v_mps`, `a_mps2` | CarState.vEgo / aEgo (opendbc speed Kalman filter) at 100 Hz, sampled | measured |
| `steer_angle_deg`, `steer_rate_degps`, `steer_pressed`, `gas_pressed`, `brake_pressed`, `standstill` | CarState | measured (the rate is 0 on brands without a CAN rate signal, exactly as in the training logs of those brands) |
| `steer_torque_norm` | CarState.steeringTorque / 95th percentile of \|torque\| over moving steer_pressed samples of the route batch (`common.torque_norm`, >= 50 samples, else absent) | `torque=route-q95` / `torque=absent` (RAV4 sample: hands-off cruise, absent; Bolt: 42 samples, absent) |
| `das_enabled` | CarState.cruiseState.enabled OR pandaStates[0].controlsAllowed (10 Hz, held 0.3 s) | approximate (openpilot's own engaged flag is not logged); the two agree on the RAV4 and Rivian samples (99.8 %) and the Bolt (35.3 %) |
| `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2` | kinematic bicycle model v tan(angle / steerRatio) / wheelbase with the carParams values (`common.yaw_from_steering`, `--yaw derived`, default); `--yaw measured` prefers CarState.yawRate where a brand fills it (VW, Ford, Hyundai CAN; units unverified, not default); `--yaw none` leaves the group absent | `yaw=derived` (training used the IMU / pose yaw) |
| lead group | radar cars: `RadarInterface` tracks at 20 Hz -> nearest in-path track (\|y - curv d^2/2\| < 1.8 m + 1 cm/m, track seen >= 3 radar samples, `measured`, far stationary clutter rejected) -> held 0.75 s -> `common.lead_channels` (speed, episodes, 1-s slope, THW, TTC). Radar-less Hyundai / Kia / Genesis: stock ACC object of `SCC11` (CAN) / `SCC_CONTROL` (CAN-FD): `ObjValid`, 0.5 < `ACC_ObjDist` < 200 m, \|`ACC_ObjRelSpd`\| < 30 m/s (204.7 m / full-scale speed are the no-object sentinels). Other vision-only platforms: `lead_valid = 0`, lead values absent | `lead=radar` / `lead=scc` / `lead=none` |

## Validation on the five sample segments (`--dataset ccs_sample`)

5 segments -> 4 windows; the Ioniq 5 segment is dropped by the moving rule (17.5 % moving, 67 % standstill), as the
lab pipeline would. Per segment (6,000 CAN packets, 60.0 s, canValid 0.99-1.00):

| platform | v range m/s | pedals gas / brake | steer range deg | steer_pressed | cruise / controlsAllowed | lead | torque scale | window |
|---|---|---|---|---|---|---|---|---|
| Hyundai Ioniq 5 (CAN-FD, vision-only) | 0.0-2.5 | 0.00 / 0.49 | -191 .. +255 | 0.14 | 0.00 / 0.00 | none (SCC_CONTROL object = sentinels for the whole minute) | absent | no (stationary) |
| Tesla Model 3 (2025 port) | -1.6-13.3 | 0.87 / 0.00 | -235 .. +254 | 0.32 | 0.00 / 0.00 | none | 2.65 | yes |
| Rivian R1 | 0.0-34.8 | 0.00 / 0.00 | -1.3 .. +5.2 | 0.78 | 1.00 / 1.00 | none | 1.94 | yes |
| Toyota RAV4 TSS2 (radar) | 0.0-30.2 | 0.00 / 0.00 | -1.1 .. +2.2 | 0.00 | 1.00 / 1.00 | radar, valid 99.7 %: gap 26-112 m, v_rel -2.7..+3.2 m/s, THW 0.9-3.7 s, lead speed 27-33 m/s | absent (no driver steering) | yes |
| Chevrolet Bolt EUV (vision-only) | 0.0-12.9 | 0.72 / 0.03 | -5.4 .. +2.3 | 0.07 | 0.35 / 0.35 | none | absent (42 samples) | yes |

Channels present: all 20 on the RAV4 except `steer_torque_norm`; the vision-only cars lack the six lead values (lead_valid
= 0) and, where the scale is missing, the torque channel. Tensor ranges over the four windows: v -1.6..34.8 m/s, a
-2.1..+2.2 m/s^2, steering -235..+254 deg, derived yaw -0.53..+0.24 rad/s, torque_norm -1.1..+0.9. The files
`$EXT/ccs_sample/work/windows.npz` + `meta.parquet` (contract) and the copies `$EXT/ccs/work/sample_windows.npz` +
`sample_meta.parquet` are the scorer's consistency target. `score.py --dataset ccs_sample` runs end to end: single
windows, so only a pipeline check (size: RAV4 -> SMALL 0.86 and Model 3 -> MID 0.65 right, Bolt EUV -> MID 0.71 and
Rivian -> MID 0.43 vs LARGE 0.33 wrong; powertrain: the three BEVs 0.87-0.99 BEV, the RAV4 0.83 BEV wrong).

Runtime: 1.2 s CPU for the CAN replay of one segment (pure-Python parser, 0.2 ms per packet, 55 MB RSS), 1.7-3.7 s wall
per segment once a worker is warm (radar cars at the upper end), 14 s for a worker's first segment of a brand; a
process start costs 1-4 minutes cold on the loaded machine. Full list: 60,619 segments ~ 40 CPU-hours -> 5-6 h with 6
workers (`nice -n 10`); the budget list (49,563) ~ 4.5 h. Memory: X is 2.9 GB float32 for the full list.

## Known gaps

1. Yaw, curvature and lateral acceleration are steering-derived (no slip, no roll; training used the IMU / pose yaw);
   expect a systematic deviation at high lateral acceleration. Sensitivity check: score the same windows with
   `--mask-channels yaw_rate_radps,curv_1pm,lat_acc_mps2` or decode with `--yaw none`.
2. `das_enabled` is "stock cruise engaged or panda controls allowed", not the controller's own flag; on openpilot
   longitudinal cars controlsAllowed is a good proxy, on stock-cruise cars the two coincide by construction.
3. Lead on radar cars comes from a simple in-path gate on the radar tracks without the vision fusion of radard
   (training's `leadOne`); it is more permissive (static objects in the path survive until the 60-m rule) and misses
   leads in sharp curves. Vision-only platforms (Tesla, Rivian, Bolt, E-GMP Hyundai / Kia / Genesis, VW, Subaru,
   Nissan, Mazda, radar-less Honda) carry no lead at all, while in training the camera model gave them one: the
   powertrain decision on most BEVs runs without lead features (CANDIDATES.md caveat). The stock-ACC fallback for
   Hyundai / Kia / Genesis is implemented but unverified: the only CAN-FD sample carried sentinels; the sign of
   `ACC_ObjRelSpd` is assumed to be lead-minus-ego; `segments_qc.csv` records `scc_sign_corr` (correlation of d gap/dt
   with the reported relative speed; positive confirms the sign) and `--scc-sign -1` flips it, `--no-scc-lead` disables it.
4. The torque scale is per route batch (the downloaded segments of a route, 1-5 for most seen platforms), not the
   whole drive as in training: a weaker normalisation, and absent whenever the minute(s) hold fewer than 50 moving
   steer_pressed samples (hands-off cruise segments).
5. One `vehicle_id` = one device (pseudonymous sha1 of the dongle id); a device moved between two cars of the same
   platform is one vehicle, a car with two devices is two. `drive_id` = device x route; segments of a route that were
   not selected leave holes in the drive (allowed: the gap flag sits on the first row of a block).
6. Segments dropped: no `carParams`, no CarInterface in opendbc 0.3.1 (24 segments of the budget list, see "Decode
   result"), canValid < 0.5, any stale grid sample (fewer than 600 fresh CarState samples), moving fraction < 0.2.
   `decode_summary.json` counts the reasons (without the platform of a skipped segment; the per-platform attribution
   below is the difference between the selected and the decoded counts).
7. Labels: 149 of the 172 selected platforms come from the lab table, 23 (2,706 segments) from `specs_external.csv`
   (public specs, +-0.05 m^2, two of them `unknown` powertrain: Honda Freed, Toyota Avalon TSS2); MIXED platforms
   (gasoline + hybrid merged upstream, e.g. Ford Explorer, Lexus NX TSS2, RAV4 TSS2 2023) are `unknown` for the
   powertrain task, as in training. Selected segments by label: SMALL 29,806 / MID 18,212 / LARGE 12,601; BEV 5,602,
   ICE 30,563, HEV 17,330, PHEV 2,488, FCEV 299, unknown 4,337.
8. Identifiers: `select.txt`, `select_budget.txt`, `select_summary.json`, `shared_dongles.txt` and `database.json`
   stay in `$EXT/ccs/` (device and route ids); the decoder writes pseudonyms only; the two markdown reports carry counts.

## Decode result (budget list, 10 Nov 2026)

Run: `ccs_decode.py --list <external>/ccs/select_budget.txt --dataset ccs --workers 6` on the owner's download (49,570
files, 74 GB); wall time 11,405 s (3.2 h), 17.9 CPU-hours of CAN replay (1.37 s per segment); `--yaw derived`, stock-ACC
lead on, sign +1 (median `scc_sign_corr` +0.70 over the 4,702 stock-ACC windows with a value, 4,745 decoded segments: the assumed lead-minus-ego sign holds).

Accounting: 49,563 segments found in 23,302 routes -> 47,006 decoded (`segments_qc.csv`, one row each) -> 1,243 dropped
by the moving rule (moving fraction < 0.2; all of them below 0.2, median 0.0) -> 45,763 windows (762.7 h) from 4,061
devices, 21,987 drives, 113 car models, 161 platforms (172 selected). Skipped before decoding: 2,557 segments.

| reason | segments | where |
|---|---|---|
| `canValid` below 0.5 | 2,460 (2,457 at exactly 0.00; one each at 0.03, 0.35, 0.40) | nine Honda platforms lose every segment: HONDA_CIVIC 400, HONDA_CIVIC_BOSCH 400, HONDA_CIVIC_2022 400, HONDA_CRV_5G 400, HONDA_FIT 282, HONDA_CRV 106, HONDA_HRV 100, HONDA_HRV_3G 60, HONDA_FREED 50 (2,198; the other Honda / Acura platforms decode normally, so it is a DBC / parser mismatch of those platforms in opendbc 0.3.1, not a CAN-bus problem); partial losses RAM_HD_5TH_GEN 64 of 143, CHRYSLER_PACIFICA_2018 42 of 389, SKODA_OCTAVIA_MK3 31 of 322, TOYOTA_PRIUS 14 of 400, the rest 1-12 per platform |
| stale 10 Hz grid samples (lab gap rule) | 73 (1-570 stale samples per segment, median 24) | spread over the platforms |
| no CarInterface in opendbc 0.3.1 | 24 | HYUNDAI_TUCSON_4TH_GEN 20 (the platform's other 345 segments decode, so these 20 carry a logged fingerprint that folds to a name without an interface), HONDA_CRV_HYBRID_6G 3, HYUNDAI_SONATA_HEV_2024 1 |

Consequences for the labels: the Honda Civic and Fit (seen SMALL ICE), the HR-V (held-out SMALL ICE) and the Freed
(absent) are missing from the set; the CR-V is present only through HONDA_CRV_HYBRID (13 devices); strata 70 / 29 / 14
car models instead of the selected 72 / 30 / 15.

Per-platform quality (`segments_qc.csv`, decoded segments; the per-platform rows of `decode_summary.json` hold the same
means):

* `can_valid_frac` 0.81-1.00, mean 0.992; the lowest platform means are the Toyota / Lexus TSS2 ports (0.983-0.986).
* Lead source: radar on 15,949 segments (Toyota / Lexus 12,483 windows, Honda 1,286, Ford 915, older Hyundai 794,
  Rivian 61), `lead_valid_frac` mean 0.63 (0.27 on KIA_NIRO_PHEV, 0.38 ACURA_ILX, 0.48 FORD_ESCAPE_MK4 at the low end,
  0.73-0.80 on HYUNDAI_SONATA, LEXUS_LC_TSS2, TOYOTA_MIRAI, TOYOTA_PRIUS_V at the high end); stock-ACC object on 4,804
  segments of 33 Hyundai / Kia / Genesis platforms, `lead_valid_frac` mean 0.69; none on 26,253 (every vision-only brand:
  Tesla, Rivian for the most part, GM, Subaru, Nissan, Mazda, Volkswagen group, Chrysler, radar-less Honda, E-GMP). A
  segment of a stock-ACC or radar platform in which no object was ever valid is written as `lead=none` in `meta.note`
  although the route batch carries the lead channels (1,271 windows: 928 on stock-ACC platforms, 343 on radar platforms).
* Yaw: `yaw_source` = derived on every segment; `CarState.yawRate` is non-zero only on Ford (98.1 % of its samples,
  4.4 % of the windows), so `--yaw measured` would change Ford only.
* CAN steering rate: non-zero on 97 % of the samples on Rivian, 65 % Tesla, 56 % GM, 44-48 % Honda / Mazda / Chrysler /
  Subaru, 38 % Toyota, 23 % Hyundai, 16 % Volkswagen group, 0 % Ford and Nissan (the rate channel is 0 there, as in the
  lab's logs of those brands).
* Torque: `torque_nonzero` 0.97 overall, lowest on Mazda (0.84) and Ford (0.93); the route-batch scale is absent on
  7,434 decoded segments (15.8 %; 7,039 of the windows), evenly over the strata (14.4-15.6 %).
* Automation and domain (windows): `das_frac` mean 0.39 (`cruise_frac` 0.39, `controls_allowed_frac` 0.38), 39.3 % of
  the windows engaged for most of the minute; mean speed 18.0 m/s, moving share 0.93, standstill 5.7 %, gas pressed
  39 %, brake 13 %, driver steering 22 % of the samples; brand means of the speed 16.7 (Volkswagen group) to 19.5 m/s (GM).
* Routes: 22,238 routes with decoded segments, 1-49 segments each (median 1, 75th percentile 2; the seen stratum was
  selected in pieces of at most 5), so the drive level of the scorer sits close to the window level.
* Devices: 4,061 pseudonyms; 91 appear under more than one platform (88 under two, 3 under three: a device moved between cars, or a fingerprint change along
  the device's history), 60 of them under two size classes; `score.py` gives such a device its majority label.

## Scoring and tables

Once `ccs_decode.py` has written `$EXT/ccs/work/windows.npz` + `meta.parquet`, the scoring campaign and the collector run
exactly as for the other sets (`source env.sh` first; nothing here touches the decoder's outputs other than adding the
`probs__*` / `score__*` files next to them):

```bash
bash experiments/transfer/run_scoring.sh ccs            # the nine scoring commands below (nice -n 10, GPU when free; BOOTSTRAP=1000 by default)
$CPY experiments/transfer/collect_scores.py             # -> RESULTS_TABLES.md + results_summary.json (section "commaCarSegments (`ccs`)", key out["ccs"]["tables"])
$CPY experiments/transfer/ccs_selftest.py               # synthetic check of the tables on <external>/_selftest_ccs (see below)
```

Run list of the `ccs` case (= `RUNS["ccs_full"]` in `collect_scores.py`; the probability / score file tag in brackets):

| run | command | file tag | role |
|---|---|---|---|
| full | `score.py --dataset ccs` | `split_platform__*__full` | primary for both tasks (all 20 channels are present) |
| V1 / V2 / V3 | `--ckpts "transfer__*__V1__s*"` etc. | `transfer__*__Vk__full` | reduced-input variants for comparison with the other sets |
| full+mask-V1 / V2 / V3 | `--mask-channels V1` etc. | `split_platform__*__mask-Vk` | full models with the preset masked (secondary; internal reference exists) |
| full+mask-yaw | `--mask-channels yaw` | `split_platform__*__mask-yaw` | sensitivity: yaw / curvature / lateral acceleration are steering-derived on most brands here (no internal reference: the reference column shows `-`) |
| full+mask-lead | `--mask-channels lead` | `split_platform__*__mask-lead` | sensitivity: vision-only platforms carry no lead (no internal reference) |

Tables written by `collect_scores.ccs_tables` (all with pseudonymous ids only; every table is skipped while
`<external>/ccs/work` has no `meta.parquet` or no full-model probabilities, so the collector keeps running before the
decode is finished) and the question each one answers:

| table | answers |
|---|---|
| decisions per platform | per platform (sorted seen / held-out / absent, then footprint): label sources, lead source shares, devices / drives / windows, automation share, the full-model size and powertrain decisions with devices right and drive accuracy, the V1 / V3 (size) and V3 / V2 (powertrain) decisions; unlabelled powertrains (HEV / PHEV / FCEV / MIXED) show the BEV probability only |
| pooled per stratum | every run x stratum (all / seen / held-out / absent) x level (window / drive / vehicle = device / model): accuracy, macro-F1, recall and units right per class, chance; the vehicle and model levels carry 95 % vehicle-cluster bootstrap intervals (`CCS_BOOTSTRAP` = 300 resamples through `score.py level_eval` on the stratum subset) |
| held-out models against the internal per-model result | for each of our held-out test models present: external devices right and model-level decision with P(true) next to the internal `split_platform` result (parsed from `internal_per_model.md`, column `full`) - the "same model, new fleet" comparison |
| seen and absent models | "seen platform, new device": model-level hits and the median per-model device accuracy over the seen models, the ten weakest seen models, and every absent model |
| lead source and automation state per stratum | radar / stock-ACC / no lead and human (das_frac < 0.5) / engaged windows: window and device accuracy for both tasks (does the lead or the automation state carry the decision?) |
| evidence per vehicle | device accuracy by labelled windows per device (1, 2-4, 5-9, 10+): how many minutes of a new device the decision needs |
| predicted-share confusion per stratum | size confusion as predicted shares per true class for the full and the yaw-masked full model (does the model use the whole label space, and how much rests on the steering-derived yaw group?) |
| unlabelled powertrains | predicted BEV share per HEV / PHEV / FCEV / MIXED platform (full model); the standard per-dataset section also renders score.py's unlabelled table for the primary run |

The pooled-decision scopes and the automation split of the cross-dataset section include `ccs` as a further scope
(`PHASE3`); the "all sets" scope now covers it as well. `ccs_selftest.py` tiles the four `ccs_sample` windows into 14
synthetic platforms (all strata, both label sources, three lead sources, labelled and unlabelled powertrains, borderline
footprints, 45 devices with 1-12 windows and multi-window drives) under `<external>/_selftest_ccs/work`, runs the nine
scoring commands on it, renders the tables and checks that every table is present, that the strata rows sum to the
`all` rows, that units right never exceed the support, that the pooled `all` rows equal the `score__*.json` files, that
no long hex token appears and that the collector's `main()` still runs (into a folder under `$D/tmp`, never into the
repository). The rendered tables land in `$D/tmp/ccs_selftest/ccs_tables_selftest.md`.
