# commaCarSegments pipeline (selection, decoding, windows)

Status 10 Nov 2026 (decode and scoring of the budget list done; see "Decode result" below; the nine Honda platforms the decode lost were recovered the same day as the second phase-3 set `ccs_honda`, see "Honda re-decode"; the same evening the decoder was run on the 2018 comma2k19 logs of two seen cars, see "comma2k19"). Three scripts in `experiments/transfer/`: `ccs_platforms.py` (key folding + labels, shared),
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
9. Understeer bias of the derived yaw (found on comma2k19, whose 2018 logs carry a `carState.yawRate`; that field is the
   output of 2018 openpilot's single-track vehicle model, not an inertial sensor - `understeer_fit.py`'s `yaw_is_model_test`
   reproduces it from the same steering angle and speed to a relative error of 3e-8, UNDERSTEER.md section 9 - so the ratios
   below compare two models, and the evidence that the lab's models see the same bias is the lab's own IMU / pose yaw, on
   which the 1 + k v^2 shape appears on 78 platforms; the bias concerns `ccs`, `ccs_honda` and `comma2k19` alike): the
   bicycle-model yaw of gap 1 matches the logged one at low
   speed and overestimates it with speed, ratio derived / logged 0.98 / 0.87 at 1-5 m/s (RAV4 / Civic), 1.13 / 1.03 at
   10-15, 1.54 / 1.53 at 20-25, 1.85 / 1.78 at 25-30, 2.06 / 2.01 at 30-40 (correlation 0.999, same sign), i.e. about
   1 + k v^2 with k ~ 0.001 s^2/m^2, the understeer gradient the kinematic model ignores. Yaw, curvature and lateral
   acceleration are therefore 1.5-2x too large on highway windows relative to the lab's IMU / pose yaw. Remedy: the
   `--yaw derived-understeer --understeer-k K` option and the re-decode of the three sets as `ccs_us`, `ccs_honda_us`,
   `comma2k19_us` (section "Understeer-corrected yaw" below, with the result; RESULTS.md 2.15 and 3.7). Verified by the
   review with `comma2k19_yaw_check.py` (lab venv, `$PY`) over all 381 windows: per-bin medians within 0.03 of these values, slopes
   1.00 / 1.13 / 1.54 / 1.82 / 2.08 (RAV4) and 0.88 / 1.02 / 1.47 / 1.76 / 2.07 (Civic), k 0.0009-0.0010; the bicycle model
   recomputed from the logged steering angle and speed gives the same ratios (`yaw_check.json`, `yaw_check_all.json` in the work
   folder).
10. Sample 0 of every window is the all-zero `CarState` of the fresh 0.3.1 parser (lazy message registration): `v` = 0 at
   k = 0 in every window of the three sets (2,132 + 45,763 + 381 windows), the speed-filter start-up transient sits at
   k <= 2, and the first grid sample of a segment should be dropped or back-filled in a future decode; the lab's own logs
   do not carry this artefact. Related: the lead gate admits occasional oncoming traffic (relative speed below -30 m/s on
   1.2 % of the valid lead samples of `ccs_honda` and on fewer than 0.1 % of comma2k19's, lead speed below -2 m/s on 2.2 % /
   0.3 %, minimum -63 m/s); a `lead_speed >= 0` rule would remove it.

## Decode result (budget list, 10 Nov 2026)

Run: `ccs_decode.py --list <external>/ccs/select_budget.txt --dataset ccs --workers 6` on the owner's download (49,570
files, 74 GB); wall time 11,405 s (3.2 h), 17.9 CPU-hours of CAN replay (1.37 s per segment); `--yaw derived`, stock-ACC
lead on, sign +1 (median `scc_sign_corr` +0.70 over the 4,702 stock-ACC windows with a value, 4,745 decoded segments: the assumed lead-minus-ego sign holds).

Accounting: 49,563 segments found in 23,302 routes -> 47,006 decoded (`segments_qc.csv`, one row each) -> 1,243 dropped
by the moving rule (moving fraction < 0.2; all of them below 0.2, median 0.0) -> 45,763 windows (762.7 h) from 4,061
devices, 21,987 drives, 113 car models, 161 platforms (172 selected). Skipped before decoding: 2,557 segments.

| reason | segments | where |
|---|---|---|
| `canValid` below 0.5 | 2,460 (2,457 at exactly 0.00; one each at 0.03, 0.35, 0.40) | nine Honda platforms lose every segment: HONDA_CIVIC 400, HONDA_CIVIC_BOSCH 400, HONDA_CIVIC_2022 400, HONDA_CRV_5G 400, HONDA_FIT 282, HONDA_CRV 106, HONDA_HRV 100, HONDA_HRV_3G 60, HONDA_FREED 50 (2,198; the other Honda / Acura platforms decode normally; cause and recovery in "Honda re-decode" below); partial losses RAM_HD_5TH_GEN 64 of 143, CHRYSLER_PACIFICA_2018 42 of 389, SKODA_OCTAVIA_MK3 31 of 322, TOYOTA_PRIUS 14 of 400, the rest 1-12 per platform |
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

## Honda re-decode (`ccs_honda`, 10 Nov 2026)

**Cause.** opendbc 0.3.1's Honda `CarState` reads the gearbox message named by `CarParams.transmissionType` (`cvt` ->
`GEARBOX_CVT` 0x191, anything else -> `GEARBOX_AUTO` 0x1A3), and its pure-Python `CANParser` registers messages lazily and
reports `can_valid = False` for the whole segment while one registered message has never arrived. 0.3.1 sets
`transmissionType` from the live fingerprint, but the decoder uses the logged `carParams`, and the logging openpilot
versions never set the field for Honda (`unknown` on 2,095 of the 2,198 segments, `cvt` on 103). The traditional-CVT
platforms (Civic 2016 / Bosch / 2022, Fit, HR-V, HR-V 3G, CR-V 2016, CR-V 5G, Freed) broadcast 0x191 only, so the parser waited
for 0x1A3 forever; the hybrid Civic 2022 (103 segments, 9 devices, logged `cvt`) broadcasts 0x1A3 only and failed the other
way round. Odyssey, Pilot, Accord 2.0T / hybrid, Insight, CR-V Hybrid, Ridgeline, RDX, ILX, Honda e are automatics / eCVT /
direct drive (0x1A3 = the default) and decoded normally. Everything else was checked and is right: the DBC (every 0.3.1 Honda
DBC defines both messages), the bus layout (Nidec / radarless pt 0, Bosch-with-radar pt 1, camera 2), the fingerprint fold, the
flags, and every other 100 Hz message arrives on the expected bus (the only invalid parser message on every failing segment was
the gearbox one). The diagnosis also prototyped the 2024-08 openpilot checkout's car stack (C++ parser) as a second backend: it
decodes 20 of the 21 probe segments (not the hybrid Civic 2022, whose legacy DBC pins the gearbox to 0x191), gives bit-identical
driver signals, but applies `wheelSpeedFactor` to the wheel speeds (0.3.1 applies it to the low-speed term only), would need a
second interpreter in the pipeline and would lose the 103 hybrid segments; it was not adopted.

**Fix (chosen).** `ccs_decode.py --honda-gearbox-from-can`: before the interfaces are built, 0.3.1's own `_get_params`
rule is applied to the pt-bus addresses of the log (0x191 present and not an Acura RDX -> `cvt`; 0x1A3, or 0x191 on the RDX ->
`automatic`; neither -> `manual`) and written into the `CarParams` reader; the result is recorded per segment as
`gearbox=<cvt|automatic|manual>` in `meta.note` and as the `gearbox` column of `segments_qc.csv` (Honda only), and
`decode_summary.json` carries `honda_gearbox_from_can` / `gearbox_counts`. Nothing else changes: same parser, grid, channel
derivations, lead gate, torque normalisation, `Labeller` and `P.pseudonym` calls, so a device keeps its `ccs_v_` id in both
sets. Without the option the decoder is byte-identical to the phase-3 run (checked: the original and the edited script on the
20 check segments give the same sha1 for `windows.npz` and `meta.parquet`, equal `segments_qc.csv` and `decode_summary.json` up
to the timing columns).

```bash
# lists (data disk only; ids never enter the repository)
$CPY experiments/transfer/ccs_honda_list.py --out $EXT/ccs/select_honda.txt           # the nine platforms within select_budget.txt: 2,198 lines
$CPY experiments/transfer/ccs_honda_list.py --platforms HONDA_ODYSSEY HONDA_PILOT HONDA_ACCORD --limit 20 --out $EXT/ccs/ccs_honda_check.txt
# consistency check (option off vs on on the 20 check segments; optionally the legacy-stack dumps of the diagnosis)
nice -n 10 $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_honda_check.py --segments-root $EXT/ccs/segments \
    --list $EXT/ccs/ccs_honda_check.txt [--legacy-dumps <scratch> --legacy-list $EXT/ccs/ccs_honda_probe.txt] --out <scratch>/ccs_honda_check.json
# decode (467 s with 6 workers), score (9 runs), collect, self-test
nice -n 10 $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_decode.py --segments-root $EXT/ccs/segments \
    --list $EXT/ccs/select_honda.txt --dataset ccs_honda --workers 6 --honda-gearbox-from-can      # log: $EXT/logs/ccs_decode_honda.log
bash experiments/transfer/run_scoring.sh ccs_honda && $CPY experiments/transfer/collect_scores.py && $CPY experiments/transfer/ccs_selftest.py
```

**Consistency (`ccs_honda_check.py`).** Part 1, option off vs on on 20 segments of platforms that decode either way (Odyssey 7,
Pilot 7, Accord 6; 20 devices, 20 routes; the rule yields `automatic` on 19 and `cvt` on one Accord): 20 / 20 decoded both
ways, 20 windows matched, every one of the 20 channels identical on the 10 Hz grid (max absolute difference 0 on the 14
continuous channels, 0.00 % differing samples on the 6 binary ones, no NaN mismatch), `can_valid` identical (min 0.998), the
`meta.note` fields identical up to the added `gearbox=` entry. Tolerance: exact equality, and it holds. Part 2, the
option (0.3.1) against the legacy car stack on the 21 probe segments of the diagnosis (two per family plus Odyssey and
Accord; per-sample CarState dumps held onto the same 10 Hz grid, 12,600 samples):

| CarState field | after 0.5 s (start-up transient excluded): differing samples, max / p99 abs diff | whole minute |
|---|---|---|
| steering angle | 77.7 % differ by float rounding: max 2.4e-5 deg, p99 6.1e-6 deg | sample 0 is all-zero in 0.3.1 (lazy registration): max 235.8 deg at that one sample |
| steering rate, steering torque, derived yaw | 0 differing samples | 0.07 % / 0.16 % (sample 0 only) |
| gas, brake, steer_pressed, standstill, cruise enabled / available | 0.000 % differing | 0.02-0.13 % (sample 0 only) |
| vEgo, aEgo on the 15 pairs without `wheelSpeedFactor` | median \|dv\| 2e-7 m/s, p99 < 1e-4 m/s, isolated maxima 0.011 m/s and 0.13 m/s^2 (one sample each) | same |
| vEgo, aEgo on the 6 pairs with `wheelSpeedFactor` 1.025 (CR-V 5G, HR-V, CR-V 2016) | legacy vEgo = 1.025 x the 0.3.1 vEgo at speed (median ratio 1.025 on every pair; max \|dv\| 0.47 m/s, \|da\| 0.20 m/s^2): the two stacks apply the factor differently; `ccs` carries the 0.3.1 convention, so the re-decode keeps it | same |
| can_valid | 0.998-1.000 with the option; 0.954-0.999 legacy (stricter fixed-list checks), 0.000 on the hybrid Civic 2022 pair | - |

Tolerances for part 2: driver and flag signals exact after the transient (holds), speed / acceleration within 1e-3 m/s and
1e-2 m/s^2 at the 99th percentile on the cars without the factor (holds; the isolated maxima are single-sample filter
events), and the factor cars differing by the documented 2.5 % only (holds). The option therefore reproduces the legacy stack
up to the one convention that `ccs` already carries.

**Decode accounting.** 2,198 segments found in 931 routes (230 devices) -> 2,189 decoded (`segments_qc.csv`) -> 57 dropped by the
moving rule (moving fraction 0-0.175) -> 2,132 windows (35.5 h) from 230 devices, 913 drives, 5 car models, 9 platforms;
wall time 467 s with 6 workers (1.3 s CPU per segment, as in the main run). Gearbox rule: `cvt` 2,086, `automatic` 103 (all of
them HONDA_CIVIC_2022, 9 devices: the hybrid Civic, which also broadcasts 0x184). Skipped 9: `canValid` 0.00 on 5 (one Fit
segment, four CR-V 5G segments; 3 devices) where `WHEEL_SPEEDS` 0x1D0 and `VSA_STATUS` 0x1A4 never appear on the powertrain
bus although the gearbox message does (a wiring / bus matter of those devices, not the gearbox rule), and 1 segment each with
17 / 28 / 30 / 31 stale grid samples (56.5-58.0 s of CAN). `can_valid_frac` mean 0.999 on every platform (min 0.967). Per
platform (segments in the list / decoded / windows / devices / drives; lead source and its valid share; scale = share of
segments whose route batch has no torque scale; DAS % = mean engaged share of the windows):

| platform | model (label) | stratum | in list | decoded | windows | devices | drives | lead | DAS % | scale absent |
|---|---|---|---|---|---|---|---|---|---|---|
| HONDA_CIVIC | Honda Civic (SMALL ICE) | seen | 400 | 397 | 385 | 33 | 186 | radar 97.5 % of the segments, valid 0.71 | 46 | 13 % |
| HONDA_CIVIC_BOSCH | Honda Civic (SMALL ICE) | seen | 400 | 400 | 395 | 55 | 154 | none (radarless) | 40 | 10 % |
| HONDA_CIVIC_2022 | Honda Civic (SMALL ICE; 103 segments / 9 devices are the hybrid, `gearbox=automatic`) | seen | 400 | 400 | 393 | 36 | 148 | none (radarless) | 46 | 16 % |
| HONDA_CRV_5G | Honda CR-V (SMALL ICE) | seen | 400 | 396 | 380 | 92 | 154 | none (radarless) | 45 | 16 % |
| HONDA_FIT | Honda Fit (SMALL ICE) | seen | 282 | 281 | 273 | 7 | 116 | radar 99 %, valid 0.70 | 35 | 23 % |
| HONDA_CRV | Honda CR-V (SMALL ICE) | seen | 106 | 106 | 106 | 4 | 46 | radar 89 %, valid 0.39 | 44 | 100 % (steer_pressed is never set on this port: torque above the threshold is not reached) |
| HONDA_HRV | Honda HR-V (SMALL ICE) | held-out | 100 | 99 | 97 | 2 | 62 | radar 99 %, valid 0.66 | 20 | 9 % |
| HONDA_HRV_3G | Honda HR-V (SMALL ICE) | held-out | 60 | 60 | 55 | 3 | 17 | none (radarless) | 56 | 22 % |
| HONDA_FREED | Honda Freed (SMALL, powertrain unknown; public spec) | absent | 50 | 50 | 48 | 1 | 30 | radar 100 %, valid 0.77 | 26 | 26 % |

Same `meta.parquet` columns and note fields as `ccs` plus the `gearbox=` entry; strata 1,932 seen / 152 held-out / 48 absent
windows; 9 of the 230 devices also appear in `ccs` under another platform (same pseudonym); 2 devices appear under more than one
platform inside the set (one of them, a possible hybrid, under three). Caveat for the labels: HONDA_CIVIC_2022 carries the lab's ICE label for the whole platform while
opendbc files the 2025-26 Civic Hybrid under the same name; the 103 `gearbox=automatic` segments (9 devices) are most likely
that hybrid and are marked in `meta.note` for the write-up to treat (a per-device split or an `unknown` powertrain for them).

**Scoring.** `run_scoring.sh ccs_honda` = the nine runs of the `ccs` case; `collect_scores.py` registers `ccs_honda` as a
second phase-3 set (`PHASE3`): its own `ccs_tables` section, and the phase-3 scope of the pooled decisions, the automation
split and the cross-dataset summary include it. Full model, all 2,132 windows: size SMALL for every platform (device level
226 / 230 = 98.3 % [96.5, 99.6], model level 5 / 5; drive level 88.2 %, window level 83.8 %); powertrain ICE for every
labelled platform (227 / 229 devices = 99.1 % [97.8, 100.0], 4 / 4 models, drive 94.1 %, window 89.3 %); the Freed
(unlabelled) is called ICE with mean P(BEV) 0.20. Held-out HR-V: 5 / 5 devices right in both tasks (model-level P(SMALL)
0.73, P(ICE) 0.84) against the internal result of 46.0 % drive accuracy / SMALL (0.50) and 95.0 % / ICE (0.84). The
reduced-input variants fall off for size (V1 47.8 %, V2 73.0 %, V3 86.5 % of the devices) and barely for powertrain (V1
86.0 %, V2 97.8 %, V3 98.3 %); the lead mask changes nothing (vision-only platforms dominate), the yaw mask collapses the size
decision (14.3 % of the devices: on these small cars the size call rests on the steering-derived yaw group). Engaged windows
score lower than human-driven ones (size 73.7 % vs 91.1 %, powertrain 80.7 % vs 95.6 % window accuracy).


## comma2k19 (`comma2k19`, 10 Nov 2026, evening)

**What the set is.** comma2k19 (comma.ai, 2018, MIT licence): two of its chunks, downloaded by the owner. Chunk 1 is one
device on a 2017 Toyota RAV4 with Toyota Safety Sense P (opendbc platform `TOYOTA_RAV4`, 188 one-minute segments), Chunk 3
one device on a 2016 Honda Civic Touring (`HONDA_CIVIC`, a traditional-CVT Nidec car, 202 segments); both are seen SMALL ICE
models of the lab table. The point of the set: the decoder on six-year-older logs (2018 openpilot, 2018 hardware), and a
"seen platform, new device, old software" probe; it is reported as its own small set (`OLD_SOFTWARE` in `collect_scores.py`),
not pooled into the commaCarSegments device scope. The raw logs (one minute of 2018 rlog each: `can` on buses 0 and 1 plus
the panda's transmit echoes 128 / 129, carState / controlsState / carControl at 100 Hz, radarState and liveTracks at 20 Hz,
`pandaStateDEPRECATED` at 1 Hz, no unknown events) are untrusted downloads: nothing in them is executed, the decoder reads
them with the same private parser as `ccs`.

**Layout and options** (all off by default; the default run is byte-identical to the `ccs` / `ccs_honda` decoder: checked on
the five verification segments, same sha1 of `windows.npz` and `meta.parquet` for the script before and after the change,
`segments_qc.csv` and `decode_summary.json` equal up to the timing columns):

```bash
nice -n 10 $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_decode.py --segments-root $EXT/comma2k19/segments \
    --layout comma2k19 --fingerprint-map Chunk_1=TOYOTA_RAV4,Chunk_3=HONDA_CIVIC --das-from controlsState \
    --ignore-absent-messages --dataset comma2k19 --root $EXT --workers 6      # log: $EXT/logs/ccs_decode_comma2k19.log (153 s)
bash experiments/transfer/run_scoring.sh comma2k19 && $CPY experiments/transfer/collect_scores.py && $CPY experiments/transfer/ccs_selftest.py
```

| option | what it does | why |
|---|---|---|
| `--layout comma2k19` | segments at `<root>/Chunk_N/<dongle>\|<route>/<seg>/raw_log.bz2`; dongle = the part before `\|`, route = the whole folder name; `.bz2` read like `.zst` | the chunk layout of the public archive |
| `--fingerprint-map Chunk_1=TOYOTA_RAV4,Chunk_3=HONDA_CIVIC` | for every segment under a mapped folder the `CarParams` is built the way openpilot builds it at start-up: opendbc 0.3.1 `CarInterface.get_params(platform, fingerprint, car_fw=[], alpha_long=False, is_release=False, docs=False)` (its exact 0.3.1 signature) from the mapped platform and the segment's own live fingerprint (address -> payload length per bus, buses below 128, as `can_fingerprint` collects it); sets the specs, flags, safety configuration and, by 0.3.1's own Honda rule (0x191 on the pt bus and not an RDX -> `cvt`), `transmissionType`, so the Civic decodes without the Honda gearbox option | 2018 openpilot logged `carParams` in a route's first segment only and the chunks hold middle segments: none of the 390 segments carries one. Provenance is uniform by construction (`cp=constructed` in `meta.note`, `cp_source` in `segments_qc.csv`); a logged `carParams`, if one ever appears, is only compared with the constructed one (`cp_logged_diff`: fingerprint fold, transmissionType, safety model / param, wheelbase, steerRatio, mass) |
| `--das-from controlsState` | `das_enabled = controlsState.enabled OR cruiseState.enabled`, `controlsState` (100 Hz) held onto the grid with the 0.3 s rule like `pandaStates`; `das=controlsState;ctrl_enabled=` in `meta.note`, `controls_enabled_frac` / `das_source` in `segments_qc.csv` (`ctrl_allowed` stays 0: the 1 Hz `pandaStateDEPRECATED` is not read) | the lab definition of `das_enabled`; `pandaStates` does not exist in 2018 logs |
| `--ignore-absent-messages` | a registered CarState parser message whose address never appears on the parser's bus in the segment's live fingerprint is marked `ignore_alive` (the parser's own rule for messages registered without a frequency, e.g. `BLINKERS_STATE`), also for the messages the CarState registers lazily during the replay; checksum, counter and timeout checks of every message the log carries stay in force; the waived names go to `segments_qc.csv` (`absent_msgs`) | 2018 openpilot logged no camera bus and the 2017 RAV4 sends no `PRE_COLLISION` 0x283 on bus 0 (in 2018 openpilot sent that message itself: it sits in the `sendcan` events and the bus-128 echo, never among the received bus-0 frames), while 0.3.1 registers the camera-bus HUD messages (Toyota `LKAS_HUD`; Honda `ACC_HUD`, `LKAS_HUD`, `BRAKE_COMMAND`) and `PRE_COLLISION`, and holds `canValid` at 0 while a registered message has never arrived (the Honda gearbox case again: without the option every segment is dropped with `canValid` 0.00). The waived messages feed stockAeb / stockFcw / HUD pass-through / cruiseState.speed only, none of the 20 channels; with the pt parser valid, the Toyota CarState also takes its accurate torque-sensor steering angle path, as in the `ccs` decode |

Everything else is the `ccs` code path: grid, channel derivations, radar lead gate (both cars have an opendbc `RadarInterface`:
TSS-P radar on bus 1, Nidec radar on bus 1), torque normalisation per route batch, windows, `Labeller` and `P.pseudonym`
(the two devices get `ccs_v_` ids like every other device).

**Accounting.** 390 segments found in 42 route folders (21 per chunk; route sizes 1-21 segments, Chunk 1 2-15 and Chunk 3
1-21; 8 of the 42 folders have gaps in the segment numbering, 7 of them in Chunk 1, which the drive builder flags on the first
row after a gap and which cost no window) -> 387 decoded -> 6 dropped
by the moving rule -> 381 windows (6.4 h) from 2 devices, 42 drives, 2 car models; 153 s with 6 workers. Skipped 3: stale 10 Hz
grid samples (77 and 110 on two Civic segments of 52.0 / 48.7 s, 381 on one RAV4 segment of 21.5 s of CAN). `cp_source` =
constructed on all 387, no logged `carParams` to compare. Per car:

| car | label | segments | decoded | windows | drives | can_valid mean (min) | lead | DAS % | cruise / ctrl_enabled | mean speed (windows) | pedals gas / brake / steer_pressed (windows) | torque scale | waived messages | gearbox |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Toyota RAV4 2017 (TSS-P) | SMALL ICE, seen | 188 | 187 | 185 | 21 | 0.987 (0.983) | radar on 185 of 187, `lead_valid` 0.78 | 56 (57 % of the windows engaged) | 0.55 / 0.55 | 22.8 m/s | 0.26 / 0.12 / 0.15 | 192-366 per route batch, never absent | `cam:LKAS_HUD; pt:PRE_COLLISION` | - (`transmissionType` unknown, as 0.3.1 leaves it for Toyota) |
| Honda Civic 2016 (Nidec) | SMALL ICE, seen | 202 | 200 | 196 | 21 | 0.997 (0.995) | radar on 199 of 200, `lead_valid` 0.87 | 62 | 0.61 / 0.61 | 20.9 m/s | 0.17 / 0.12 / 0.16 | 2,061-3,070, never absent | `cam:ACC_HUD,LKAS_HUD,BRAKE_COMMAND` | `cvt` on all 200 (0x191 on the pt bus) |

All 20 channels present on every window. `cruise_frac` and `controls_enabled_frac` coincide on both cars (2018 openpilot
engaged with the stock cruise button). Tensor ranges: v -0.1..46.6 m/s, a -3.5..+8.0 m/s^2 (the values above 4 m/s^2 are 7
samples at k <= 2 of a window, the speed-filter start-up transient of a fresh CarInterface per segment, as in `ccs` /
`ccs_honda`; at k >= 3 the maximum is 3.6), steering -482..+385 deg, derived yaw -0.52..+0.53 rad/s, torque_norm -1.4..+2.2.

**Checks.** (a) Logged against constructed `CarParams`: no segment carries a `carParams` event (0 of 390, probed with the lab
`LogReader` as well as with the decoder's parser), so the comparison is vacuous; the constructed values are RAV4: `toyota`
safety with param 841 (EPS scale 73 | ALT_BRAKE | STOCK_LONGITUDINAL), wheelbase 2.65 m, steerRatio 16.88, mass 1,792 kg,
radar available, `transmissionType` unknown; Civic: `hondaNidec` safety, wheelbase 2.70 m, steerRatio 15.38, mass 1,462 kg,
radar available, `transmissionType` cvt. (b) One window per car (the window with the most valid lead samples and some
steering): corr(dv/dt, a) after sample 1 = 0.60 (RAV4, a parking-lot window with 121 deg of steering) / 0.92 (Civic); over the
whole set, from sample 3 on, median 0.92 (10th percentile 0.61; `ccs_honda` 0.98 / 0.79); yaw sign = steering sign on 100 % of
the moving samples with |angle| > 2 deg in both windows and in every one of the 371 windows with enough such samples
(corr(yaw, angle) 0.98 / 0.99); curv - yaw / v and lat_acc - v yaw at most 3e-9 1/m and 4e-7 m/s^2; THW - d / v at most
4e-7 s over the valid lead samples (lead 1.2-47.5 m, relative speed -8.1..+4.8 m/s, lead_speed = v + v_rel to 5e-7); gas and
brake never both pressed (0 of 228,600 samples of the set), standstill never with v > 0.5 m/s. (c) Decoder grid against the
LOGGED 2018 `carState` of the same segment (five segments per car spread over the chunk, lab `LogReader` under `$PY`,
nearest-time match, at most 12 ms apart; sample 0 excluded):

| signal | RAV4: max / p99 of \|difference\| | Civic: max / p99 | what differs and why |
|---|---|---|---|
| vEgo | 0.05-0.22 / 0.03-0.11 m/s | 0.09-0.17 / 0.04-0.09 m/s | no speed scaling: the median ratio decoder / logged is 1.0000 on every segment and both equal the mean wheel speed (ratio 1.000); the residual is the speed Kalman filter (different gains then and now) and the ZOH-vs-nearest sampling of a 100 Hz signal |
| aEgo | 0.57-1.72 / 0.33-1.16 m/s^2 | 0.96-2.69 / 0.35-1.07 m/s^2 | the 2018 `aEgo` is noisier (its standard deviation 1.1-1.6 x the decoder's) and lags the decoder's by 2-3 grid samples (0.2-0.3 s); against the finite-difference acceleration of the logged speed the decoder's `aEgo` correlates 0.78-0.98, the logged one 0.82-0.96; corr(decoder, logged) 0.64-0.94: the speed-filter gains changed between 2018 openpilot and opendbc 0.3.1, and the training logs carry the modern filter |
| steering angle | 0.2-0.4 / 0.2-0.3 deg, median offset 0.00 | 0.2-0.4 / 0.1-0.2 deg | corr 0.9996-1.0000; the differences are the sensor resolution and sampling (on the RAV4 the decoder uses the torque-sensor angle minus a learned offset, 2018 openpilot the 1.5 deg angle sensor plus its fraction) |
| steering rate | 7-16 / 2-7 deg/s | 7-53 / 2-12 deg/s | corr 0.97-0.99 (RAV4) / 0.59-0.98 (Civic): a quantised 100 Hz signal sampled at different instants; no scale or sign difference |
| steering torque | 37-63 / 19-40 units | 2,116-5,870 / 1,092-5,083 units | RAV4: sampling only (corr 0.82-0.999, same units, same `steer_pressed` threshold of 100). Civic: corr -0.977 to -0.999, i.e. **the sign is flipped**: opendbc later inverted the sign of Honda `STEER_TORQUE_SENSOR` so that positive torque turns left like on the other brands; the decoder applies today's DBC to the 2018 bytes and therefore carries the convention of the training logs, the 2018 `carState` the old one (`steer_pressed`, from \|torque\| > 1,200, agrees) |
| gas / brake / steer_pressed / standstill | differ on 0.0-0.7 / 0.0 / 0.0-1.2 / 0.0 % of the samples | 0.0-0.2 / 0.0 / 0.0-2.2 / 0.0-2.0 % | same thresholds (Toyota gas > 0, Honda `steer_pressed` 1,200, Toyota 100; brake from the brake switch); the differing samples sit at the pedal transitions (ZOH against nearest sample) |
| das_enabled against controlsState.enabled OR cruiseState.enabled of the log | 0.0-0.7 % | 0.0 % | same definition, sampled differently |

**Scoring** (`run_scoring.sh comma2k19` = the nine runs of the `ccs` case; 381 windows, 42 drives, 2 devices, 2 models).
Powertrain, full model: ICE on both cars (device 2 / 2, model 2 / 2; drive level 88.1 % [85.7, 90.5], window 70.9 %; Civic
P(ICE) 0.71, RAV4 0.68); V3 speed-only variant: ICE on both (drive 83.3 %, window 64.3 %); every variant and mask calls both
cars ICE. Size, full model: RAV4 -> SMALL (P 0.54; drive 57.1 %, window 55.7 %), Civic -> MID by a hair (P(SMALL) 0.485 against
P(MID) 0.491; drive 52.4 %, window 48.0 %), i.e. device level 1 / 2, window level 51.7 %, drive level 54.8 %; the V3 speed-only
variant gets both right (device 2 / 2, drive 97.6 %, window 82.2 %), and so does the full model under the V3 mask (window
96.6 %, drive 100 %), while the yaw mask collapses the size call to LARGE (window 3.4 %), as on `ccs_honda`. Automation split:
the full powertrain model is right on 100 % of the human-driven windows (155; mean P(BEV) 0.06) and on 50.9 % of the engaged
ones (226; mean P(BEV) 0.47). A plausible reading, to be kept as an observation: on both cars 2018 openpilot did the
longitudinal control itself (`sendcan` holds BRAKE_COMMAND 0x1FA on the Civic and ACC_CONTROL 0x343 on the RAV4 in every probed
segment, although today's CarParams files the TSS-P RAV4 under stock longitudinal), so the engaged minutes carry a 2018 controller the models never saw, and the full models lean on
exactly those channels (lead, acceleration), whereas the speed-only variant does not. The collector adds the set's per-dataset
section and per-model rows, a pooled scope row of its own and the two devices to the "all sets" scope, and the two automation
rows; every pre-existing table is unchanged except those pooled totals (checked by diff against the previous
`RESULTS_TABLES.md`); `ccs_selftest.py` passes.

## Understeer-corrected yaw (`ccs_us`, `ccs_honda_us`, `comma2k19_us`)

**Formula.** The default `--yaw derived` is the kinematic bicycle model, yaw = v tan(angle / steerRatio) / wheelbase (known gap
9: it overestimates the lab's IMU / pose yaw, and the yaw 2018 openpilot's vehicle model logged on comma2k19, by about 1 + k v^2). `--yaw derived-understeer --understeer-k K` divides it by the
linear single-track steady-state factor,

    yaw_us = v tan(steer_angle / steerRatio) / (wheelbase (1 + K v^2)),      K in s^2/m^2

with ONE global K passed on the command line (fitted on the lab's own IMU / pose yaw - the truth the models were trained on -
and validated on comma2k19; never per platform, so that no platform-specific lab calibration enters the external features), then
curv = yaw_us / v and lat_acc = v yaw_us exactly as before (`common.curvature_and_lat_acc`). `tan` versus the small-angle form is
kept as in the default path (`common.yaw_from_steering`); `K <= 0` or a missing value is refused, and `--understeer-k` without the
mode is refused too. Everything else is untouched: grid, CarState hold, lead gate (whose in-path test keeps the uncorrected bicycle
curvature on purpose, so that the lead channels stay identical), torque scale per route batch, labels and pseudonyms; a re-decode
therefore differs from the original set in the three yaw-group channels only, which the collector verifies (below). Bookkeeping:
`yaw_source = derived-us` in `segments_qc.csv`, `yaw=derived-us;k=K` in `meta.note` (the `k=` field follows the `yaw=` field),
and `yaw = derived-understeer`, `understeer_k`, `yaw_formula`, `yaw_sources` in `decode_summary.json`. The default path is
byte-identical to the previous script: the five verification segments decoded with the edited script give the same sha1 for
`windows.npz` and `meta.parquet` as `$EXT/ccs_sample/work/` and the copies `$EXT/ccs/work/sample_windows.npz` /
`sample_meta.parquet` (checked 10 Nov 2026).

**Fit of K.** The global value is K = 0.00079289 s^2/m^2 (hours-weighted median of the one-parameter k over the 62 train /
validation lab platforms, 423.8 h; `understeer_fit.py`, numbers in `understeer_fit.json`); method, spread, per-platform table,
comma2k19 validation and caveats in `UNDERSTEER.md`. The three re-decodes use exactly this K.

**Smoke (K = 0.001, two comma2k19 segments per car, the fastest and a mixed-speed one).** Corrected and original yaw against the
logged 2018 `carState.yawRate` matched to the decoder grid by nearest time (sample 0 excluded; least-squares slope through the
origin and median of the sample ratios over |yawRate| > 0.02 rad/s): Civic 20-25 m/s original 1.48 / corrected 0.96, 25-30
1.85 / 1.00, 30-40 2.29 / 1.03 (1-5 m/s 0.88 / 0.87: the low-speed shortfall is not a speed effect); RAV4 25-30 m/s 1.89 / 1.03,
30-40 2.17 / 1.04; correlation 0.999-1.000 in every bin; `yaw_us = yaw / (1 + K v^2)` holds to 1e-8 against the original windows.
In that smoke only `steer_torque_norm` differed from the original besides the yaw group, because one segment per route was
decoded and the torque scale is a route-batch quantity: the real re-decodes use the same lists as their originals, so the
batches, and with them every non-yaw channel, are the same.

**Commands** (`source env.sh`; `K` = the fitted global value; the flags are exactly those of the original decodes, see "Decode
result": `ccs` ran with `--list select_budget.txt --dataset ccs --workers 6`, i.e. `--yaw derived`, stock-ACC lead on with sign
+1, `--honda-gearbox-from-can` OFF, layout `ccs`, `--das-from panda`; `ccs_honda` the same with `--list select_honda.txt` and
`--honda-gearbox-from-can`; `comma2k19` with the three comma2k19 options and no list):

```bash
# ccs_us: the budget list, gearbox option off as in the original ccs decode (ignore the 'canValid 0.00' Honda skips, they are ccs_honda's)
nice -n 10 $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_decode.py --segments-root $EXT/ccs/segments \
    --list $EXT/ccs/select_budget.txt --dataset ccs_us --workers 6 --yaw derived-understeer --understeer-k K      # ~3.2 h; log: $EXT/logs/ccs_decode_us.log
# ccs_honda_us: the nine Honda platforms with the gearbox option, as the original ccs_honda decode
nice -n 10 $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_decode.py --segments-root $EXT/ccs/segments \
    --list $EXT/ccs/select_honda.txt --dataset ccs_honda_us --workers 6 --honda-gearbox-from-can --yaw derived-understeer --understeer-k K
# comma2k19_us: the 2018 logs with the comma2k19 options
nice -n 10 $D/ext/venv_opendbc/bin/python3 experiments/transfer/ccs_decode.py --segments-root $EXT/comma2k19/segments \
    --layout comma2k19 --fingerprint-map Chunk_1=TOYOTA_RAV4,Chunk_3=HONDA_CIVIC --das-from controlsState --ignore-absent-messages \
    --dataset comma2k19_us --root $EXT --workers 6 --yaw derived-understeer --understeer-k K
# score (the nine runs of the ccs case per set), collect, self-test
bash experiments/transfer/run_scoring.sh ccs_us ccs_honda_us comma2k19_us
$CPY experiments/transfer/collect_scores.py && $CPY experiments/transfer/ccs_selftest.py
```

**Tables (`collect_scores.yaw_correction_tables`, section "Understeer-corrected yaw re-decodes" at the end of
`RESULTS_TABLES.md`, numbers in `results_summary.json` under `yaw_correction`).** The three `*_us` sets are registered with the
kind `ccs_us` (the nine runs of `ccs_full`) and are kept out of every pre-existing table (per-dataset sections, cross-dataset
summary, named-vehicle rows, every pooled-decision scope including "all sets", the automation split, `ccs_tables`), so the
earlier output is unchanged. Per pair (`ccs_us` / `ccs`, `ccs_honda_us` / `ccs_honda`, `comma2k19_us` / `comma2k19`):

| table | content |
|---|---|
| alignment check | same `vehicle_id` / `drive_id` / `w_idx` rows in the same order and the same label columns; max \|diff\| of every non-yaw channel = 0 in float32 with the same NaN pattern; the yaw group differs; the meta columns and the notes (after the `yaw=` / `k=` fields) equal. A misaligned pair gets one `ABORTED` line and no tables |
| per stratum | all / seen / held-out / absent x window / device / model level, for the full model and V1, both tasks: accuracy, macro-F1, per-class recall and units right per class, original -> corrected with the delta |
| platform decision flips | full model, size and powertrain: platforms whose model-level decision (mean probability over the platform's labelled windows -> arg-max) changed, with P(true) and devices right before / after; MID / LARGE platforms recovered (MISS -> OK) or lost (OK -> MISS); devices right before / after per class |
| yaw-masked reference | the `full+mask-yaw` rows of both decodes (the mask hides the only channels that changed, so the probabilities must agree; the collector reports whether they do) |
| summary | one line per set: device accuracy and model-level / platform hits before -> after for both tasks, and whether the masked runs agree |

`ccs_selftest.py` builds `_selftest_ccs_us` (the synthetic set with the yaw group divided by 1 + 0.001 v^2 and nothing else
changed), scores it with the nine runs, checks the alignment result, the read-back `k`, the delta arithmetic, the strata sums,
the flip bookkeeping and the masked-run identity, verifies that two misaligned copies (rows permuted; one speed sample changed)
are refused, and runs the collector twice (with and without the synthetic pair registered): everything before the understeer
section must be byte-identical.

**Result (11 Nov 2026; RESULTS.md 2.15 and 3.7, the full tables in the collector section).** The three re-decodes ran with
K = 0.00079289 (`ccs_us` 10,444 s = 2.9 h, `ccs_honda_us` 591 s, `comma2k19_us` 184 s, 6 workers) with the same accounting as
their originals (`ccs_us` 49,563 found -> 47,006 decoded / 2,557 skipped -> 45,763 windows; 2,132 and 381 windows on the other
two); the alignment check passed on all three (same rows and labels, every non-yaw channel with max |diff| 0, the yaw-masked
runs of both decodes identical) and `ccs_selftest.py` passes with the pair. `ccs_us` against `ccs`, full models: size device
accuracy 54.1 -> 63.2 % (+9.1; macro-F1 46.4 -> 60.3; the lab's internal held-out vehicle reference is 63.5 / 59.7), models
63 -> 72 of 113, platforms right 95 -> 108 of 161 (27 changed decision: LARGE recovered 15 / lost 0, MID recovered 4 / lost 2,
SMALL lost 4, 2 LARGE wrong -> wrong), devices right SMALL 1,606 -> 1,594 of 1,762, MID 351 -> 442 of 1,274, LARGE 241 -> 532 of
1,025 (the four Pacifica platforms, Palisade, CX-9, Sienna 4th gen, Rivian R1, Maverick, Ranger, G80, Hyundai Genesis, both
Escalades and the Crafter come back; the Honda Odyssey / Pilot / Ridgeline, Carnival, Atlas, Explorer and G90 / GV80 do not);
powertrain device accuracy 87.8 -> 88.9 % (+1.1), models 86 -> 85 of 88 (the F-150 Lightning at P(BEV) 0.52 -> 0.46), devices
BEV 538 -> 539 of 621, ICE 1,667 -> 1,693 of 1,891. `ccs_honda_us` against `ccs_honda`: size 98.3 -> 93.5 % of 230 devices
(SMALL 226 -> 215: 10 devices to MID, 2 to LARGE, one back from MID; no platform changed), powertrain 99.1 -> 99.6 %. `comma2k19_us` against `comma2k19`: no
car-level decision changes (size 1 of 2, powertrain 2 of 2). The correction is post hoc (motivated by the comma2k19 ratio,
fitted on the lab's own IMU / pose yaw, never tuned on the external labels); the original decodes stay the as-run results of
RESULTS.md 2.12-2.14 and `--yaw derived-understeer` is the recommended option for future external decodes; the `*_us` sets
stay out of every pooled table. Remaining: the +15 % residual at 30-40 m/s, the geometry scale a of a third of the platforms
(UNDERSTEER.md section 5), the sample-0 and oncoming-lead artefacts of gap 10.

## Scoring and tables

Once `ccs_decode.py` has written `$EXT/ccs/work/windows.npz` + `meta.parquet`, the scoring campaign and the collector run
exactly as for the other sets (`source env.sh` first; nothing here touches the decoder's outputs other than adding the
`probs__*` / `score__*` files next to them):

```bash
bash experiments/transfer/run_scoring.sh ccs            # the nine scoring commands below (nice -n 10, GPU when free; BOOTSTRAP=1000 by default); `ccs_honda` and `comma2k19` take the same case
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

The pooled-decision scopes and the automation split of the cross-dataset section include `ccs` and `ccs_honda` as a further scope
(`PHASE3`, summed over both sets); the "all sets" scope covers them as well, and `ccs_tables` is rendered once per phase-3 set
(`comma2k19` is a scope of its own, `OLD_SOFTWARE`, without `ccs_tables`: two cars need no eight tables). `ccs_selftest.py` tiles the four `ccs_sample` windows into 14
synthetic platforms (all strata, both label sources, three lead sources, labelled and unlabelled powertrains, borderline
footprints, 45 devices with 1-12 windows and multi-window drives) under `<external>/_selftest_ccs/work`, runs the nine
scoring commands on it, renders the tables and checks that every table is present, that the strata rows sum to the
`all` rows, that units right never exceed the support, that the pooled `all` rows equal the `score__*.json` files, that
no long hex token appears and that the collector's `main()` still runs (into a folder under `$D/tmp`, never into the
repository). The rendered tables land in `$D/tmp/ccs_selftest/ccs_tables_selftest.md`.
