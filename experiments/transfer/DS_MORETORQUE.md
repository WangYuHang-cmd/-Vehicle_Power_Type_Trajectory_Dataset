# MoreTorque/rlogs adapter (`ds_moretorque.py`): community openpilot logs -> VTT transfer windows

Adapter `experiments/transfer/ds_moretorque.py` (contract: `common.py` / `CONTRACT.md`), written 9 Nov 2026. Source: the four
platform folders of the Hugging Face dataset `MoreTorque/rlogs` (Apache-2.0) that the owner downloaded to
`<data-root>/external/moretorque/` (75 GB, 7,636 one-minute openpilot `rlog.bz2` segments): Hyundai Sonata
2020, Mazda 3 2019, Mazda CX-50 and Omoda E5. Role in the campaign (CANDIDATES.md 2.5, RESULTS.md section 4): the only
external set besides commaCarSegments with **all 20 training channels from the lab's own sources** (carState, a device
IMU / localizer yaw, a vision lead, driver torque and pedal flags), so a V0 test with measured lateral channels; SMALL and
MID models only; one BEV (Omoda E5, one device) as a powertrain sanity probe; one seen model (Sonata) next to three models
absent from the lab inventory, all from community openpilot forks.

Outputs (`<data-root>/external/moretorque/work/`, `<external>/moretorque/work/` below): `windows.npz` + `meta.parquet` + `windows_summary.json`
(the contract files, written by `common.save_windows`), `drives10hz/<drive_id>.parquet` (the 10 Hz table of every drive,
26 signal columns incl. diagnostics), `drives_index.parquet`, `segments_qc.csv`, `drives_qc.csv`, `vehicles_qc.csv`,
`decode_summary.json`, `validation.json`, `report_tables.md`, `compare_training.md` (channel statistics against a
4,000-window sample of the training split, `training_reference_stats.json`), `labels.json`, `logs/`. No device id, route
name or path appears in any of them (pseudonyms `veh_<sha1('vtt-transfer' + device)[:8]>` and
`drv_<sha1('vtt-transfer' + device/route/first segment)[:10]>`).

Run (openpilot venv, `source env.sh`):

```
nice -n 10 $PY -I experiments/transfer/ds_moretorque.py --smoke 20 --dataset moretorque_smoke   # 5 consecutive segments per folder
nice -n 10 $PY -I experiments/transfer/ds_moretorque.py --workers 6                              # everything (stage decode + assemble + compare)
$PY -I experiments/transfer/ds_moretorque.py --stage assemble [--steer-rate can|derived] [--torque keep] [--pedals keep] [--das any] [--align run]
```

The decode stage is the only heavy part (worker pool, at most 6 processes, `nice` inherited); the assemble stage re-cuts
windows from the saved 10 Hz tables in about a minute, so the channel options can be varied without decoding again.

## 1. Source data

Layout: `<FOLDER>/<16-hex device>/<route>--<segment>--rlog.bz2` (two route-name styles: a date-time stamp for three
folders, the new 8-hex / 10-hex openpilot style for the Omoda). 7,636 files of 5-15 MB (one < 1 MB, 23 of 1-5 MB), 12
devices, 476 routes; the dataset keeps only part of every route, so the files form 1,226 runs of consecutive segment
numbers. A run is one **drive** here (the unit of the torque scale and of the window blocks). Recording months from the
file names: Sonata July-September 2024 (1,495 / 1,524 / 12 segments), Mazda 3 November 2023 (205), June-September 2024
(7 / 1,046 / 1,146 / 3) and 90 undated, CX-50 July-August 2024 (859 / 335), Omoda all undated (914). No GPS and no
wall-clock in the logs.

| folder | devices | routes | runs (drives) | segments | on disk |
|---|---|---|---|---|---|
| HYUNDAI SONATA 2020 | 1 | 125 | 313 | 3,031 | 33 GB |
| MAZDA 3 2019 | 7 | 189 | 434 | 2,497 | 23 GB |
| MAZDA CX-50 | 3 | 107 | 380 | 1,194 | 13 GB |
| OMODA_E5 | 1 | 55 | 99 | 914 | 7.5 GB |
| total | 12 | 476 | 1,226 | 7,636 | 75 GB |

Runs are short: median 2 segments per drive, 90th percentile 17, longest 110 (1.8 h); 450 drives are a single segment.
267 segments hold fewer than 5,900 carState messages (route starts and ends; the shortest 17.4 s), every segment has
`canValid` on >= 51 % of its samples (median 100 %).

Topics in the logs (every folder, from the probes and the full decode): `carState` 100 Hz, `carControl` 100 Hz, `can` /
`sendcan` 100 Hz, `modelV2` 20 Hz, `liveLocationKalman` 20 Hz, `liveParameters` 20 Hz, `longitudinalPlan` 20 Hz,
`liveTorqueParameters` 4 Hz, `liveCalibration` 4 Hz, `carParams` once per segment (`lateralPlanDEPRECATED` on the Sonata
and Mazdas, `carOutput` on the Omoda). Not retained by the dataset: `controlsState` / `selfdriveState`, `radarState`,
`livePose`, `pandaStates`, `deviceState`, `initData` (no software version), `gpsLocation*`. The logs were read with the
cereal LogReader of the lab's openpilot checkout (19 Aug 2024, `tools/lib/logreader.py`, bz2 and zst), which knows every
union member these 2024 logs carry (0 unknown events).

`carParams` of the four folders (identical on every decoded segment of a folder): `HYUNDAI SONATA 2020` (carName hyundai,
wheelbase 2.840 m, steer ratio 15.3, mass 1,649 kg), `MAZDA 3 2019` (mazda, 2.725 m, 16.5-17.0, 1,497 kg), `MAZDA CX-50`
(mazda, 2.814 m, 15.5, 1,667 kg), `OMODA_E5` (chery, 2.630-2.750 m, 18.0, 1,921-2,086 kg; the fork changed its guesses
over time). `radarUnavailable` is True on all four (vision lead, as on the lab's radar-less Hyundais and Mazdas),
`pcmCruise` True (openpilot engages through the stock cruise, so the controller flag and `cruiseState.enabled` coincide).

## 2. Timeline and channel mapping

The lab built its public tables from qlogs: row k = the k-th `carState` sample (every 10th message of the 100 Hz stream),
real sample times only for the gap rule (`dt > 0.3 s`), every other topic put onto the carState times by a causal zero-order
hold with a maximum age (0.3 s controls, 0.625 s pose, 0.75 s lead). The adapter reproduces this from the rlogs:

* **carState** 100 Hz -> every 10th message, the phase running across segment boundaries of a drive (the stream is
  continuous: boundary dt 0.098-0.101 s, median dt 0.100 s in every drive; only 6 of the 1,226 drives contain a gap row,
  the largest 6.1 s, inside a short segment); `t_s` = real seconds from the drive's first sample; windows are the lab's
  fixed 600-row blocks from the drive start (`--align drive`).
* **other topics** decimated to their qlog rates the same way before the hold (`carControl` every 10th = 10 Hz like
  `controlsState`, `liveLocationKalman` every 5th = 4 Hz, `modelV2` leads every 5th = the 4 Hz of `radarState` in a qlog), so
  the held yaw and lead have the training staircase texture; `--native-rates` holds the 20 / 100 Hz streams directly.

Status: *measured* = same quantity and source as training; *derived* = computed with the lab's definition or a documented
substitute; *absent* = NaN (training "dropped group" encoding).

| channel | source | status | notes |
|---|---|---|---|
| v_mps | carState.vEgo | measured | |
| a_mps2 | carState.aEgo | measured | the vehicle's own Kalman acceleration, as in training |
| steer_angle_deg | carState.steeringAngleDeg | measured | left-positive (opendbc convention) |
| steer_rate_degps | carState.steeringRateDeg (Sonata) / `common.smooth_derivative` of the 10 Hz angle (Mazda 3, CX-50, Omoda) | measured / derived | the three fork platforms never populate the CAN rate (0 on every sample while the wheel moves), whereas the lab's own Mazda drives carry a live rate (non-zero on 36-69 % of samples), so the derived substitute is closer to training than the dead zeros; `--steer-rate can` keeps the zeros, `derived` forces the substitute everywhere; decided per vehicle (`rate_alive` = CAN rate non-zero on >= 0.5 % of moving samples) |
| steer_torque_norm | carState.steeringTorque / 95th percentile of \|torque\| over moving & steer_pressed samples of the drive (>= 50 samples) | measured (lab rule) | raw units differ by fork (Mazda fork +-9,000 counts, Sonata +-500), irrelevant after the per-drive scale; **Omoda: absent** - the fork reports an unsigned torque (0..87, never negative while the wheel turns both ways), so the group is dropped (`torque=absent-unsigned`; `--torque keep` keeps the magnitude) |
| yaw_rate_radps | -liveLocationKalman.angularVelocityCalibrated.z, held 0.625 s | measured | the lab used `angularVelocityDevice.z` with the same sign (POSE_YAW_SIGN = -1); that field is empty in these logs, the calibrated frame differs from the device frame by the mount pitch / yaw only; `angularVelocityCalibrated.z` is right-positive in openpilot (corr with the steering angle -0.83 to -0.99 on the probes), so the lab sign gives a left-positive yaw; yaw_valid = calibrated.valid & inputsOK |
| curv_1pm, lat_acc_mps2 | yaw / v while v >= 1 m/s and yaw valid; v x yaw | derived (lab definition) | |
| gas_pressed, brake_pressed | carState.gasPressed / brakePressed | measured | **Omoda: absent** - the fork never sets either flag (0 on every sample of 914 segments, 15 h, with |a| up to 3.3 m/s^2); NaN = the training "pedals dropped" state (`pedals=absent-dead`; `--pedals keep`) |
| steer_pressed | carState.steeringPressed | measured | present on all four (the Omoda fork derives it from its torque magnitude) |
| standstill | carState.standstill | measured | |
| das_enabled | carControl.enabled (held 0.3 s) OR carState.cruiseState.enabled | measured (controller flag substitute) | `controlsState` is not retained; `carControl.enabled` is the same controller "enabled" state; both flags coincide on these pcmCruise cars (disagreement <= 0.4 % of samples on ten vehicles, 2.4 % on one Mazda 3; on the Omoda the stock cruise runs without the fork being engaged on 15.9 % of samples, and das_enabled takes either). **Fork caveat:** the Sonata and CX-50 forks run lateral-only automation (`carControl.latActive` while not enabled, with real steering torque commanded: \|actuators.steer\| mean 0.15-0.19, p95 up to 1.0); under the lab definition those samples have das_enabled = 0 although the car is steering itself. The share is recorded per window as `lat_only=` in `meta.note` and per vehicle in `vehicles_qc.csv`; `--das any` folds latActive into das_enabled |
| lead_valid | modelV2.leadsV3[0] (the t = 0 vision lead), held 0.75 s: prob >= 0.5 and d > 0.5 m | measured (vision) | radard's rule for radar-less cars (status = prob > 0.5); training already holds vision leads on `radarUnavailable` cars, incl. the lab's Sonata and Mazda drives |
| lead_d_m | leadsV3[0].x[0] - 1.52 m | measured (vision) | radard subtracts RADAR_TO_CAMERA (1.52 m) when it turns a vision lead into `radarState.leadOne.dRel`; same here |
| lead_v_rel_mps, lead_speed_mps | leadsV3[0].v[0] - vEgo; v + v_rel | measured (vision) | the model's absolute lead speed minus the CAN ego speed (radard uses its own ego-speed estimate; mean difference 0.04 m/s, std 0.28 on a probe) |
| lead_a_slope_mps2, thw_s, ttc_s | `common.lead_channels` | derived (lab definitions) | causal 1-s slope inside lead episodes, THW while v > 0.5 m/s, TTC while closing > 0.1 m/s |

`channels_present` lists all 20 channels for the Sonata and the two Mazdas, 17 for the Omoda (no torque, no pedal flags).

## 3. Labels

| folder | platform (adapter) | fingerprint in carParams | model | wheelbase x mean track | footprint | class | powertrain | stratum | source |
|---|---|---|---|---|---|---|---|---|---|
| HYUNDAI SONATA 2020 | HYUNDAI_SONATA | `HYUNDAI SONATA 2020` (upstream name, folds to HYUNDAI_SONATA) | Hyundai Sonata (DN8, non-hybrid) | 2840 x 1628 mm | 4.624 m^2 | MID | ICE | **seen** (lab: 126 train + 19 validation drives of the model, 24 of this platform) | lab platform table (`vtt/specs_data/platform_specs.csv`), powertrain of the lab drives |
| MAZDA 3 2019 | MAZDA_3_2019 | `MAZDA 3 2019` (fork) | Mazda 3 (BP, 2019-) | 2725 x 1575 mm | 4.292 m^2 | SMALL | ICE | absent (brand seen: Mazda CX-5, CX-9, 6) | `specs_external.csv` row "Mazda 3 / BP 2019-" (medium confidence); the carParams wheelbase 2.725 m confirms the BP generation |
| MAZDA CX-50 | MAZDA_CX50 | `MAZDA CX-50` (fork) | Mazda CX-50 (2023-) | 2815 x 1660 mm | 4.673 m^2 | MID | ICE | absent | `specs_external.csv` (Mazda Canada sheet, high confidence); carParams wheelbase 2.814 m |
| OMODA_E5 | OMODA_E5 | `OMODA_E5` (fork) | Omoda E5 (2024-) | 2610 x 1550 mm | 4.045 m^2 | SMALL | BEV | absent | `specs_external.csv` (Chery AU / carexpert, medium confidence; evkx 2630 mm -> 4.08, SMALL either way) |

None of the four is within 0.05 m^2 of a cut (no borderline flag). The CX-50 hybrid (MY2025, Toyota system) post-dates these
July-August 2024 recordings; the North-American Mazda 3 2.5 has no mild hybrid; the Sonata fingerprint is the non-hybrid
DN8 (the hybrid is a separate openpilot platform). Stratum = `split_platform` status of the model in
`build/index/drives.parquet` through `ccs_platforms.Labeller`: the Sonata is a training model (so its numbers are an
"in-distribution model, new device, new fork" reference, not a held-out test), the other three are absent from the lab
inventory; no held-out model is present.

## 4. Counts

All 7,636 segments decoded (0 failed, 0 skipped: every segment carries the folder's fingerprint, >= 10 carState messages
and `canValid` >= 50 %). 4,549,191 ten-Hz rows = 126.4 h; 7,276 window candidates, 4 dropped for a gap, 82 for the moving
rule (< 20 % of samples at >= 1 m/s) -> **7,190 windows = 119.8 h** in 1,155 drives (71 drives give no window: 55 shorter
than 600 rows, 16 entirely stationary).

| folder | platform | model | class / powertrain | stratum | vehicles | drives | segments | windows | hours | DAS share | lateral-only share | lead sample share | windows with a lead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HYUNDAI SONATA 2020 | HYUNDAI_SONATA | Hyundai Sonata | MID / ICE | seen | 1 | 313 | 3,031 | 2,926 | 48.8 | 0.634 | 0.285 | 0.368 | 0.665 |
| MAZDA 3 2019 | MAZDA_3_2019 | Mazda 3 | SMALL / ICE | absent | 7 | 434 | 2,497 | 2,319 | 38.6 | 0.728 | 0.150 | 0.478 | 0.720 |
| MAZDA CX-50 | MAZDA_CX50 | Mazda CX-50 | MID / ICE | absent | 3 | 380 | 1,194 | 1,071 | 17.9 | 0.140 | 0.483 | 0.697 | 0.943 |
| OMODA_E5 | OMODA_E5 | Omoda E5 | SMALL / BEV | absent | 1 | 99 | 914 | 874 | 14.6 | 0.296 | 0.111 | 0.684 | 0.957 |
| total | | | MID 3,997 / SMALL 3,193; ICE 6,316 / BEV 874 | | 12 | 1,226 | 7,636 | 7,190 | 119.8 | | | | |

DAS share = mean `das_frac` of the windows (lab definition); lateral-only share = samples with `latActive` and not
`enabled`; lead sample share = mean `lead_valid`; training windows: DAS 0.54, lead 0.54, 82 % of windows with a lead.

Per vehicle (pseudonyms; `vehicles_qc.csv` has every column):

| vehicle | model | drives | segments | windows | 10 Hz hours | gas / brake flag alive | CAN rate alive | torque signed | yaw ~ steer corr | yaw valid | DAS | lateral-only | op long (drives) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| veh_1c1bc25c | Hyundai Sonata | 313 | 3,031 | 2,926 | 50.3 | yes / yes | yes | yes | 0.93 | 0.999 | 0.62 | 0.29 | 0 / 313 |
| veh_8a7fb9d8 | Mazda 3 | 201 | 1,324 | 1,211 | 21.9 | yes / yes | no | yes | 0.93 | 0.995 | 0.67 | 0.25 | |
| veh_95a2afec | Mazda 3 | 154 | 940 | 878 | 15.5 | yes / yes | no | yes | 0.91 | 0.996 | 0.73 | 0.00 | |
| veh_5adb5f7d | Mazda 3 | 32 | 176 | 174 | 2.9 | yes / yes | no | yes | 0.87 | 0.998 | 0.79 | 0.13 | |
| veh_1e5bb0ca | Mazda 3 | 29 | 36 | 35 | 0.6 | yes / yes | no | yes | 0.94 | 0.999 | 0.05 | 0.00 | |
| veh_8b094748 | Mazda 3 | 8 | 11 | 11 | 0.2 | yes / yes | no | yes | 0.98 | 0.999 | 0.00 | 1.00 | |
| veh_a1531e81 | Mazda 3 | 7 | 7 | 7 | 0.1 | yes / yes | no | yes | 0.94 | 0.999 | 0.26 | 0.31 | |
| veh_660276e0 | Mazda 3 | 3 | 3 | 3 | 0.05 | yes / yes (3 fires in 3 min; kept, < 10 min of evidence) | no | yes | 0.97 | 0.999 | 0.97 | 0.00 | |
| Mazda 3 together | | 434 | 2,497 | 2,319 | 41.3 | | | | | | 0.68 | 0.15 | 348 / 434 |
| veh_f20695f6 | Mazda CX-50 | 123 | 634 | 555 | 10.4 | yes / yes | no | yes | 0.93 | 0.992 | 0.02 | 0.80 | |
| veh_f1e45994 | Mazda CX-50 | 236 | 434 | 419 | 7.2 | yes / yes | no | yes | 0.92 | 0.996 | 0.32 | 0.00 | |
| veh_d632de27 | Mazda CX-50 | 21 | 126 | 97 | 2.1 | yes / yes | no | yes | 0.92 | 0.999 | 0.01 | 0.59 | |
| Mazda CX-50 together | | 380 | 1,194 | 1,071 | 19.7 | | | | | | 0.13 | 0.48 | 20 / 380 |
| veh_93d1a56f | Omoda E5 | 99 | 914 | 874 | 15.1 | **no / no** (0 fires in 15 h) | no | **no** (0..127) | 0.87 | 0.957 | 0.30 | 0.11 | 40 / 99 |

"op long" = drives whose `carParams.openpilotLongitudinalControl` is set (the fork, not the stock ACC, controlled speed
while engaged): most Mazda 3 drives, a few CX-50 and Omoda drives, never the Sonata.

Torque group per drive (lab rule, `drives_qc.csv`): `drive-q95` on 887 drives, `absent-few-samples` (fewer than 50
moving & steering samples, mostly single-segment drives) on 240 (Sonata 66, Mazda 3 126, CX-50 48), `absent-unsigned` on
the 99 Omoda drives; the NaN share of `steer_torque_norm` in the windows is 3.6 % (Sonata), 9.3 % (Mazda 3), 6.5 %
(CX-50), 100 % (Omoda). Steering rate: CAN on the Sonata, derived on the 913 fork drives. Pedal flags: present everywhere
except the Omoda.

## 5. Validation (`validation.json`, `report_tables.md`, `compare_training.md`)

Smoke test first (`--smoke 20`: 5 consecutive segments of one drive per folder, 6 s with 4 workers): speeds 0-34 m/s,
pedal shares 0.0-0.94 per segment, steering -442..+448 deg (parking), yaw ~ steering correlation +0.92 to +0.99 per
drive, vision lead on 0.7-40 % of samples, boundary dt 0.098-0.101 s, all 18 contract checks passed; the same checks
on the full set:

* **Contract checks (18 / 18 ok):** flags 0 / 1 only; every lead value NaN exactly where `lead_valid` = 0 and `lead_valid`
  never NaN; `lead_speed = v + v_rel`; curvature NaN below 1 m/s; every window >= 20 % moving; yaw left-positive
  (correlation with the steering angle +0.87 to +0.98 for all 12 vehicles, the lab's own drives give +0.87 to +0.98);
  no 16-hex device token, route name or path in any output (`segments_qc.csv`, `drives_qc.csv`, `vehicles_qc.csv`,
  `drives_index.parquet`, `meta.parquet`, the drive tables, the json / md files were scanned).
* **Speed:** window means 24.7 (Sonata, mostly highway at 30-35 m/s), 21.9 (Mazda 3), 15.6 (CX-50), 9.3 m/s (Omoda,
  urban) against 17.5 in training; maxima 38.5-38.9 m/s on the three ICE cars, 32.5 on the Omoda. Moving share
  0.88-0.96 per window.
* **Acceleration (CAN aEgo):** std 0.48 / 0.59 / 0.76 / 0.38 m/s^2 (training 0.60); 1-99 % range within -2.3..+2.3
  (training -2.0..+1.8); segment extremes down to -11 m/s^2 on one Mazda 3 (CAN artefact, 1 sample, left in: the scorer
  clips).
* **Steering:** |angle| 99th percentile 216 / 218 / 299 / 328 deg (training 192); the Sonata CAN rate is an unsigned
  quantised magnitude (p01 = 0, p99 = 92 deg/s) **exactly as in the lab's own Sonata windows** (p01 = 0, p99 = 128); the
  derived rate of the forks has std 22-29 deg/s and 99 % range +-57..95, matching the lab's Mazda CAN rate (std 28,
  +-93). Torque: |norm| 95th percentile 0.62-0.77 (lab Sonata / Mazda windows 0.84 / 0.83 at the 99th, std 0.25-0.33 vs 0.26-0.27).
* **Yaw:** std 0.050 / 0.057 / 0.085 / 0.040 rad/s (training 0.066); valid on 95.7-99.9 % of samples; curvature NaN
  share 4-15 % (= standstill + invalid; training 11 %); lateral acceleration 99 % range within +-2.7 m/s^2.
* **Flags:** gas 0.19 / 0.22 / 0.50 (training 0.30; lab Sonata windows 0.49, lab Mazda 0.27), brake 0.10 / 0.06 / 0.20
  (training 0.10), steer_pressed 0.12 / 0.12 / 0.25 / 0.26 (training 0.17), standstill 0.03 / 0.05 / 0.11 / 0.06
  (training 0.08). Omoda: gas / brake absent (see section 2).
* **Automation:** `carControl.enabled` and `cruiseState.enabled` disagree on <= 0.4 % of samples on ten vehicles and on
  2.4 % on one Mazda 3; on the Omoda the stock cruise is on without the fork being engaged on 15.9 % of samples
  (das_enabled takes either). Lateral-only
  automation (`latActive` without `enabled`): Sonata 28.5 % of samples (28.6 % of its windows have it on more than half
  of their samples), CX-50 48.3 % (52.8 % of windows), Mazda 3 15.0 % (12.8 %), Omoda 11.1 % (10.1 %); per window in
  `meta.note` (`lat_only=`).
* **Lead:** vision lead on 36.8-69.7 % of samples (training 54 %), 66.5-95.7 % of windows have one (training 82 %);
  gap 1-99 % range 1.7-133 m (training 2.5-120), relative speed -8.9..+3.9 m/s (training -8.2..+4.2), THW median
  1.7-2.9 s (training mean 2.7), TTC and slope ranges like training (`compare_training.md`).
* **Labels:** fingerprints 100 % consistent per folder (3,031 / 2,497 / 1,194 / 914 segments); carParams wheelbases
  agree with the spec rows (Sonata 2.840 = 2840 mm, Mazda 3 2.725 = BP 2725 mm, CX-50 2.814 vs 2815 mm; the Omoda
  fork's 2.63-2.75 m are placeholders: Chery's own sheet says 2610 mm; SMALL in every case).

Runtime: decode stage 2,873.5 s (47.9 min) for 7,636 segments with 6 `nice -n 10` workers on a machine already running
at load ~20 (2.7 seg/s overall; 2.25 s CPU per segment, 1.7 s median, of which ~1 s is the bz2 + capnp read); the smoke
test on an idle machine ran at 3.6 seg/s with 4 workers; assembly + comparison 85 s; peak RSS per worker < 300 MB; the 10 Hz
drive tables take 203 MB, `windows.npz` 138 MB.

## 6. Caveats for the scorer and the report

1. **Driver and car are confounded**: one device for the Sonata and the Omoda, 3 and 7 for the CX-50 and Mazda 3 (two
   Mazda 3 devices carry 90 % of that folder's hours). Report per model with the vehicle-cluster bootstrap; the Omoda is
   one car and one driver (a powertrain anecdote, not a test); no interval exists for the single-vehicle models.
2. **Stratum**: the Sonata is a training model (DN8, 24 lab drives of this platform; the lab also trained on the hybrid
   and the LF generation under the same model name), so its result is "seen model, new device, community fork", not a
   held-out test. The three fork platforms are absent from the lab inventory; the Mazda brand is seen (CX-5, CX-9, 6).
   No held-out model of the headline split is present. SMALL and MID only: no LARGE class.
3. **Community forks**: (a) the Sonata and CX-50 forks drive with lateral-only automation on 29 % / 48 % of the samples,
   which the lab definition of `das_enabled` does not flag (the steering, torque and `steer_pressed` channels of those
   windows are automation-shaped while `das_enabled` = 0); use `lat_only=` in `meta.note` to split or weight, or re-cut
   with `--das any`; (b) openpilot longitudinal control was on in most Mazda 3 drives, so engaged windows there are
   fork-controlled, not stock ACC; (c) the Omoda fork reports an unsigned torque and no pedal flags (groups dropped: in
   distribution, but the Omoda windows carry 17 channels), and its carParams geometry is a placeholder.
4. **Derived steering rate** on the three fork platforms (CAN rate dead); the Sonata's CAN rate is an unsigned magnitude,
   as in training for this model. `--steer-rate can` reproduces the dead zeros for a sensitivity run.
5. **Yaw from the calibrated frame** instead of the device frame used in training (same sign, mount-angle difference
   only); qlog-rate decimation of the pose and lead streams reproduces the training staircase (`--native-rates` for the
   smooth 20 Hz version).
6. **Vision lead everywhere** (radarUnavailable on all four), as on the lab's own radar-less Hyundais and Mazdas; the
   training Sonata windows had a lead 91 % of the time against 67 % here (highway driving with free road ahead).
7. **Operating domain**: the Sonata is almost all highway (mean 24.7 m/s), the Omoda almost all urban (9.3 m/s, 15 h);
   speed and lead shares differ from training by platform, so a per-model result also reflects the roads.
8. **Dates / provenance**: July-September 2024 for the dated routes (Mazda 3 also November 2023), Omoda undated; no GPS,
   wall-clock or software version in the logs; none of the devices is in the lab library (different fleet).
9. Not done here: scoring (GPU busy; run `$CPY experiments/transfer/score.py --dataset moretorque` for the full models,
   `--mask-channels yaw` / `torque` / `pedals` for sensitivity, `--ckpts "transfer__*V1*"` for a V1 comparison), and the
   optional radar lead from the Sonata's raw CAN (SCC11), which these logs would allow.

## 7. Files

| file | content |
|---|---|
| `experiments/transfer/ds_moretorque.py` | the adapter (stages decode / assemble / compare, options above) |
| `experiments/transfer/DS_MORETORQUE.md` | this report |
| `<external>/moretorque/work/windows.npz`, `meta.parquet`, `windows_summary.json` | the window set for `score.py --dataset moretorque` (7,190 x 600 x 20) |
| `<external>/moretorque/work/drives10hz/drv_*.parquet` | 1,226 ten-Hz drive tables (26 signal columns incl. `op_enabled`, `lat_active`, `lead_prob`, raw torque) |
| `<external>/moretorque/work/drives_index.parquet`, `segments_qc.csv`, `drives_qc.csv`, `vehicles_qc.csv` | per-drive / per-segment / per-vehicle QC (pseudonyms only) |
| `<external>/moretorque/work/decode_summary.json`, `validation.json`, `report_tables.md`, `compare_training.md`, `training_reference_stats.json`, `labels.json` | summaries, checks, comparison with a 4,000-window training sample |
| `<external>/moretorque/work/logs/run_full.log`, `assemble_final.log` | run logs |
| `<external>/moretorque_smoke/work/` | the 20-segment smoke test (16 windows; can be deleted) |
