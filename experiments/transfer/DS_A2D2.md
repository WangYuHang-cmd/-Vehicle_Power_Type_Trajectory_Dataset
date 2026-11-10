# A2D2 adapter: Audi Q7 e-tron bus signals -> VTT transfer windows

Adapter `experiments/transfer/ds_a2d2.py` (contract: `common.py` / `CONTRACT.md`), written 9 Nov 2026. Source: the
three sensor-fusion bus-signal JSONs of the Audi Autonomous Driving Dataset (A2D2, CC BY-ND 4.0, fetch record in
`<external>/a2d2/raw/FETCH.md`). One recording car (Audi Q7 e-tron quattro, diesel plug-in hybrid), driven manually in
three German towns in 2018-2019. Role in the campaign (CANDIDATES.md 1, row 8): a single-vehicle **LARGE sanity probe**
for the size model; the powertrain (PHEV) is outside the BEV-vs-ICE task and is only reported as predicted shares.

Outputs (`<data-root>/external/a2d2/work/`): `windows.npz` + `meta.parquet` + `windows_summary.json`
(the contract files, written by `common.save_windows`), `series_<drive>.parquet` (the 10 Hz frame of each sequence with
the diagnostic columns `imu_ax_mps2, imu_ay_mps2, a_speed_mps2, lat_acc_yaw_mps2, gas_pct, brake_bar`), `validation.json`
(every check below, machine-readable), and the `probs__* / score__*` files of the scorer smoke test (section 7).

Run: `source env.sh && $CPY -I experiments/transfer/ds_a2d2.py` (about 20 s; `--dry-run` builds and checks without writing).
Options: `--align drive|run` (default `drive`, the lab's fixed 600-row blocks; identical here because no sequence has a
gap), `--accel speed|imu`, `--lat-acc yaw|imu`, `--steer-pressed nan|zero|one|active`, `--gas-pct`, `--brake-bar`,
`--dataset NAME` (save a variant under another name, e.g. for sensitivity runs).

## 1. Source data

Each JSON is one dict `{signal: {"unit", "values": [[timestamp_us, value], ...]}}` with 22 signals, every signal on its own
clock (Unix microseconds). The adapter uses 10 of them; times are made relative to the first `vehicle_speed` sample of the
sequence (`t_s`), the 10 Hz grid runs from the latest first sample to the earliest last sample of the eight used streams.

| stream (json key) | unit | rate | max dt | used for |
|---|---|---|---|---|
| vehicle_speed | km/h (0.01 steps) | 50 Hz | 21 ms | v_mps, a_mps2, standstill, curv, lat_acc |
| steering_angle_calculated + steering_angle_calculated_sign | deg (0.15 steps) + 0/1 | 100 Hz | 11.4 ms | steer_angle_deg, steer_rate_degps |
| angular_velocity_omega_z | deg/s (0.01 steps) | 200 Hz | 6.4 ms | yaw_rate_radps, curv, lat_acc |
| acceleration_x, acceleration_y | m/s^2 (not gravity-compensated) | 200 Hz | 6.4 ms | diagnostics only (optional `--accel imu`, `--lat-acc imu`) |
| accelerator_pedal | % (0.4 steps) | 100 Hz | 10.9 ms | gas_pressed |
| brake_pressure | bar (0.3 steps) | 100 Hz | 11.3 ms | brake_pressed |
| latitude_degree, longitude_degree | deg | 6.25 Hz | 320 ms | yaw-sign check and location only |

Not used: acceleration_z, omega_x / omega_y, pitch / roll angle, the four wheel pulse counters, lat / lon direction flags,
accelerator_pedal_gradient_sign (its meaning is undocumented and it does not track the pedal derivative). All streams
start within 16 ms of the speed stream and end within 15 ms of it (GPS 76-136 ms earlier); no stream has a gap, so
`cut_windows` sees one run per sequence and drops no window for gaps.

| drive_id (scene folder) | location (GPS) | duration | 10 Hz rows | v max | v median | moving share (v >= 1 m/s) | standstill share (v < 0.5) | window candidates | windows kept |
|---|---|---|---|---|---|---|---|---|---|
| 20180810_150607 | Gaimersheim | 525.8 s | 5,258 | 20.4 m/s | 7.9 m/s | 0.916 | 0.072 | 8 | 8 |
| 20190401_121727 | Munich | 919.6 s | 9,196 | 10.7 m/s | 2.3 m/s | 0.581 | 0.386 | 15 | 14 (1 below the 20 % moving rule) |
| 20190401_145936 | Ingolstadt | 745.2 s | 7,452 | 12.9 m/s | 4.7 m/s | 0.746 | 0.226 | 12 | 12 |
| total | | 2,190.6 s = 36.5 min | 21,906 | | | | | 35 | **34** (0.567 h) |

Low-speed urban driving throughout: no window has a mean speed above 20 m/s; the training windows average 17.5 m/s.

## 2. Channel mapping

Status: *measured* = same quantity as training; *derived* = computed from other channels with the lab's definition or a
documented substitute; *missing* = NaN (or the training "dropped group" encoding). The scorer applies the training clip and
standardisation, so all values are raw physical units.

| channel | source | 10 Hz method | status | notes and checks |
|---|---|---|---|---|
| v_mps | vehicle_speed / 3.6 | zero-order hold (sample at or before the grid time, max age 60 ms) | measured | |
| a_mps2 | 10 Hz v_mps | `common.smooth_derivative` (central difference, 0.5-s centred mean) | derived | the lab aEgo is a speed-filter output, so the speed derivative is the faithful substitute; IMU acceleration_x agrees (corr 0.97-0.99, slope 1.00-1.05, offset -0.06 m/s^2, residual std 0.17-0.19) but is rougher (high-frequency std 0.16-0.20 vs 0.10-0.11 moving) and gravity-contaminated (pitch up to 3 deg = up to 0.5 m/s^2), so it stays a diagnostic (`--accel imu`) |
| steer_angle_deg | steering_angle_calculated x (+1 if sign == 0, -1 if sign == 1) | hold (max age 30 ms) | measured | steering-wheel angle (up to 517 deg); sign bit 1 = right turn, result left-positive: correlation with the IMU yaw rate +0.95 / +0.97 / +0.96 while moving (the lab's own windows give +0.91); implied steering ratio from the bicycle model with L = 2.994 m: 17.2 / 16.5 / 16.2 (Q7 4M specification about 16) |
| steer_rate_degps | 10 Hz steer_angle_deg | `common.smooth_derivative` | derived | no CAN rate in the data; applying the same derivative to the lab's own 10 Hz angles reproduces the CAN rate distribution (slow human-only windows: \|rate\| p75 / p90 / p99 = 12 / 68 / 262 deg/s against 10 / 66 / 283 for the CAN signal) |
| steer_torque_norm | - | - | missing (NaN) | no driver torque; the torque group is a training dropout group, so NaN is in distribution |
| yaw_rate_radps | angular_velocity_omega_z x pi/180 | hold (max age 15 ms) | measured | left-positive: GPS heading change over 3-s intervals (v > 6 m/s) = -1.03 / -0.84 / -0.95 x the integrated gyro (correlation -0.94 / -0.79 / -0.77, median ratio -1.02 / -0.89 / -1.04; compass bearing is clockwise-positive, so -1 means counter-clockwise-positive = left-positive); zero-level bias at long stops -0.08 / -0.12 / -0.14 deg/s (std 0.07), left uncorrected (0.002 rad/s against a training std of 0.066) |
| curv_1pm | v, yaw | `common.curvature_and_lat_acc`: yaw / v while v >= 1 m/s, else NaN | derived (lab definition) | NaN on 27 % of the window samples = exactly the samples below 1 m/s (checked) |
| lat_acc_mps2 | v, yaw | v x yaw (lab definition, identical to the training tensors: corr 1.000 on internal windows) | derived (lab definition) | IMU acceleration_y agrees (corr 0.97 / 0.93 / 0.85, slope 1.08 / 1.07 / 1.06) with an offset of +0.18 / +0.16 / +0.04 m/s^2 from roll (gravity leakage: 9.8 x sin(1 deg) = 0.17) and residual std 0.29-0.30, so the IMU value is not used (`--lat-acc imu` for comparison) |
| gas_pressed | accelerator_pedal > 0 % | flag at 100 Hz, hold | derived | opendbc Volkswagen rule (`gasPressed = gas > 0`, Audi is a VW-group car); pressed share 0.63 / 0.41 / 0.53 per sequence; > 2 %: 0.62 / 0.40 / 0.51; > 5 %: 0.60 / 0.37 / 0.46; > 10 %: 0.52 / 0.29 / 0.23 |
| brake_pressed | brake_pressure > 0.5 bar | flag at 100 Hz, hold | derived | zero-level readings wander between -1.2 and 0 bar (0.3-bar steps), so the threshold sits two steps above zero; pressed share 0.20 / 0.48 / 0.33; > 0 bar: 0.20 / 0.49 / 0.34; > 1 bar: 0.19 / 0.47 / 0.32; > 3 bar: 0.18 / 0.44 / 0.30 (insensitive); gas and brake are never pressed together; the long brake holds at traffic lights make the share far higher than the training mean (0.10) |
| steer_pressed | - | - | missing (NaN) | no torque or hands-on signal; **not** a training dropout group, so NaN (= the standardised mean after pre-processing) is out of distribution for this one channel; `--steer-pressed zero|one|active` are diagnostic substitutes (`active` = \|steer_rate\| > 5 deg/s or \|angle\| > 15 deg, share 0.47; internal human-only windows have steer_pressed 0.33), see section 7 |
| standstill | v < 0.5 m/s | flag | derived | share 0.07 / 0.39 / 0.23 (training mean 0.08); opendbc's VW rule is `vEgoRaw == 0`, slightly stricter |
| das_enabled | constant 0 | | derived | manual driving; the bus data carry no cruise or automation flag |
| lead_valid | constant 0 via `common.apply_channel_mask("lead")` | | missing (training dropout encoding) | no radar / camera lead; 0 is the value the training augmentation used when it dropped the lead group (NaN would mean "unknown") |
| lead_d_m, lead_v_rel_mps, lead_speed_mps, lead_a_slope_mps2, thw_s, ttc_s | - | - | missing (NaN) | |

`channels_present` in the meta frame lists the eleven channels with a source (`v_mps, a_mps2, steer_angle_deg,
steer_rate_degps, yaw_rate_radps, curv_1pm, lat_acc_mps2, gas_pressed, brake_pressed, standstill, das_enabled`); the
lead mask is applied after cutting, so `lead_valid` is not listed although it holds zeros. Scoring plan (MANIFEST.md):
the full model with the lead and torque groups absent, which the data already encode; no `--mask-channels` needed.

## 3. Labels

| field | value | source |
|---|---|---|
| model | Audi Q7 e-tron | A2D2 README ("We have equipped an Audi Q7 e-tron ...") |
| platform | Audi Q7 e-tron quattro, 4M, 2016-2018 (3.0 TDI diesel plug-in hybrid) | `specs_external.csv` |
| wheelbase / track front / track rear | 2,994 / 1,679 / 1,691 mm | `specs_external.csv` (auto-data.net row cited there) |
| footprint | 2.994 x 1.685 = **5.045 m^2 -> LARGE** (0.165 above the 4.88 cut; the alternative 1,663 mm front track gives 4.99, still LARGE) | locked cuts 4.43 / 4.88 |
| borderline | no | \|footprint - cut\| >= 0.05 |
| curb mass | 2,445 kg | `specs_external.csv` |
| powertrain | **PHEV** (diesel plug-in hybrid): outside BEV vs ICE, kept as label, reported as predicted shares only | CANDIDATES.md 1 |
| vehicle_id | `a2d2-audi-q7-etron` (one physical car, so the vehicle bootstrap has one cluster and no interval) | |
| drive_id | the three A2D2 scene folder names (public) | |

## 4. Counts

34 windows = 0.567 h, 1 vehicle, 1 model, 3 drives; size_class LARGE 34, powertrain PHEV 34, borderline 0; das_frac 0 in
every window. Windows per drive 8 / 14 / 12; window moving fraction mean 0.93 / 0.60 / 0.74, minimum 0.68 / 0.235 / 0.31.
Flag shares over all window samples: gas_pressed 0.517, brake_pressed 0.351, standstill 0.246, das_enabled 0, lead_valid 0.

Continuous channels over the 34 windows (raw units; the training clip bounds are v -1..60, a +-8, angle and rate +-720,
yaw +-1.5, curvature +-0.3, lateral acceleration +-8; nothing is outside them):

| channel | min | p1 | p50 | p99 | max | mean | std | training mean / std |
|---|---|---|---|---|---|---|---|---|
| v_mps | 0.00 | 0.00 | 5.08 | 18.5 | 20.4 | 4.90 | 4.13 | 17.50 / 11.07 |
| a_mps2 | -3.46 | -2.26 | 0.00 | 1.89 | 4.09 | -0.004 | 0.685 | 0.002 / 0.590 |
| steer_angle_deg | -481.6 | -308.7 | 0.30 | 264.8 | 516.8 | -3.28 | 83.0 | 0.31 / 51.4 |
| steer_rate_degps | -437.1 | -135.0 | 0.00 | 155.6 | 511.6 | 0.09 | 41.3 | 1.73 / 29.0 |
| yaw_rate_radps | -0.550 | -0.391 | -0.002 | 0.328 | 0.547 | -0.002 | 0.106 | 0.000 / 0.066 |
| curv_1pm | -0.196 | -0.122 | 0.000 | 0.093 | 0.231 | -0.003 | 0.031 | 0.000 / 0.018 |
| lat_acc_mps2 | -4.11 | -1.83 | 0.00 | 2.69 | 3.05 | 0.031 | 0.628 | -0.006 / 0.560 |

## 5. Validation

Per-sequence checks (all in `validation.json` -> `sequences`): the stream facts of section 1; the sign and scale checks of
section 2 (steering vs yaw correlation, implied steering ratio, GPS heading regression, IMU a_x vs dv/dt, IMU a_y vs v x yaw,
zero levels during stops longer than 5 s: gyro bias -0.08 to -0.14 deg/s with std 0.07, IMU a_x -0.03 to +0.05 m/s^2 with std
0.05-0.08); the pedal threshold tables; and completeness on the grid (no NaN in any model column except curv_1pm, whose NaN
pattern equals v < 1 m/s). Post-cut checks (`validation.json` -> `windows`, the build refuses to save on a failure): shape
(34, 600, 20) float32; every declared-present channel NaN-free (curv_1pm as above); the missing channels NaN everywhere
and lead_valid 0 everywhere; the six flags strictly 0 / 1; das_enabled 0; no sample outside the physical ranges above;
moving fraction >= 0.2 in every window; one model / size / powertrain per vehicle. `common.save_windows` raised no warning.
Round trip: `common.load_windows("a2d2")` returns (34, 600, 20) and the first window equals rows 0-599 of
`series_20180810_150607.parquet`; a second build reproduces the tensor bit for bit.

Distribution against the lab's own slow human-driven windows (internal windows with mean speed < 7 m/s and no automation,
3,000 sampled; A2D2 first): speed mean 4.9 vs 3.6 m/s; \|a\| p90 1.17 vs 1.09 (std 0.69 vs 0.65); \|steering angle\| p90 110 vs
197 deg; \|steering rate\| p90 50 vs 66 deg/s; \|yaw\| p90 0.18 vs 0.18 and p99 0.42 vs 0.46 rad/s; \|lateral acceleration\| p90
0.89 vs 0.63 m/s^2; gas_pressed 0.52 vs 0.37; brake_pressed 0.35 vs 0.36; standstill 0.25 vs 0.28. The kinematic channels
sit inside the internal range; the Q7 steers less for the same yaw than the internal mix (steering geometry v x delta /
yaw = wheelbase x ratio of about 49-51 m, the range of Honda Pilot / Odyssey, Hyundai Palisade, Acura MDX and Kia Carnival
in the training fleet). The two channels the data cannot provide are where A2D2 departs from the lab: steer_pressed
(internal human-only share 0.33, here NaN) and the lead group (internal slow human-only lead_valid 0.45, here 0).

## 6. Caveats

* One car, one driver per sequence (unknown), 36 minutes, low-speed urban driving in Germany: a sanity probe, not a test
  set; the vehicle-level bootstrap has a single cluster, so no interval exists.
* PHEV: excluded from the BEV-vs-ICE score by label; the scorer reports its predicted shares.
* `steer_pressed` is the only out-of-distribution input (NaN -> standardised mean); the lead group and the torque channel are
  absent in the in-distribution encoding of the training augmentation. Section 7 shows that the size prediction moves with
  the steer_pressed treatment, so A2D2 numbers from the full model should be read with that in mind; a variant trained
  without steer_pressed (V1 minus steer_pressed) would remove the issue.
* Derived channels: a_mps2 and steer_rate_degps are smoothed derivatives (a slightly lower variance than CAN rate samples at
  short time scales), standstill is a 0.5 m/s rule, gas / brake flags are thresholds (insensitive to the threshold, tables in
  section 2); the gyro bias is not removed.
* Licence CC BY-ND 4.0: the windows are derived material for internal evaluation only; do not redistribute them.
* A stream that starts a few microseconds after the speed stream moves the grid start by one sample (Munich grid starts at
  0.1 s, one 10 Hz row lost); harmless.

## 7. Scorer smoke test and sensitivity (not a result for the paper)

Run on the saved windows with `score.py --dataset a2d2 --task both` (GPU) and, for the variants, `--device cpu` (the GPU
was occupied by another process, see issues). Files `score__*.md/.json` and `probs__*.npz` in the work folder.

| model | inputs | A2D2 Q7 e-tron prediction (ensemble mean probability SMALL / MID / LARGE) | window acc | drive | vehicle |
|---|---|---|---|---|---|
| size GRU S9, full (lead + torque absent in the data) | 20 | **SMALL** 0.852 / 0.092 / 0.056 | 0 % (seeds 0 / 0 / 2.9 %) | 0 / 3 | 0 / 1 |
| same, `--mask-channels steer` | | SMALL 0.80 / 0.17 / 0.03 | 0 % | | |
| same, `--mask-channels yaw` | | SMALL 0.84 / 0.10 / 0.06 | 2.9 % | | |
| same, `--mask-channels lateral` | | SMALL 0.90 / 0.06 / 0.04 | 0 % | | |
| same, `--mask-channels pedals` | | SMALL 0.90 / 0.04 / 0.06 | 0 % | | |
| same, `--mask-channels V2` (longitudinal only) | | SMALL 0.88 / 0.05 / 0.06 | 0 % | | |
| same, `--mask-channels V3` (speed only) | | SMALL 0.71 / 0.21 / 0.07 | 0 % | | |
| same, steer_pressed := 0 (`--steer-pressed zero`) | | SMALL 0.85 / 0.12 / 0.04 | 0 % | | |
| same, steer_pressed := 1 (`--steer-pressed one`) | | **MID** 0.03 / 0.96 / 0.01 | 0 % | | |
| same, steer_pressed := active proxy (`--steer-pressed active`) | | SMALL 0.48 / 0.35 / 0.16 | 5.9 % | | |
| variant V1 lateral + speed (`--variant V1`, 2 seeds) | 10 | SMALL 0.68 / 0.19 / 0.13 | 2.9 % | 0 / 3 | 0 / 1 |
| variant V3 speed only (`--variant V3`, 2 seeds) | 3 | MID 0.36 / 0.43 / 0.21 | 14.7 % | 0 / 3 | 0 / 1 |
| variant V3das speed + das (1 seed, diagnostic run) | 4 | LARGE 0.26 / 0.35 / 0.40 | 50 % | 1 / 3 | 1 / 1 |
| variant V3lead speed + lead group (1 seed, diagnostic run; lead absent here) | 10 | SMALL 0.45 / 0.32 / 0.23 | 5.9 % | 0 / 3 | 0 / 1 |
| BEV vs ICE ResNet B5, full | 20 | predicted shares ICE 100 % (label PHEV, unscored) | | | |
| BEV vs ICE variant V3 | 3 | ICE 85 % / BEV 15 % | | | |
| BEV vs ICE variant V1 | 10 | ICE 59 % / BEV 41 % | | | |

Reading (results-triage rule: artefact check, then mechanism, then the user decides):

1. **The LARGE sanity probe fails**: the final size model calls the Q7 SMALL with 0.85 confidence in all 34 windows, and no
   input subset of the full model turns it into LARGE. The powertrain model calls the diesel PHEV ICE, which is plausible
   and not scored.
2. **Adapter artefacts were looked for and not found**: signs (steering and yaw both left-positive, verified against GPS
   heading and against each other, the same relation as in the training tensors), scales (implied steering ratio 16-17,
   IMU accelerations agree with the derived channels at slope 1.0-1.08), ranges (nothing outside the training clip bounds),
   and distributions (section 5) are consistent; curvature and lateral acceleration use the lab's exact definitions.
3. **The decision is fragile where the data are blind**: setting the unknown steer_pressed channel to 1 flips the call to
   MID (0.96), the activity proxy spreads it (0.48 / 0.35 / 0.16), and the reduced-input variants disagree (V1 SMALL, V3 MID,
   V3das LARGE). A single vehicle cannot separate "the model does not generalise to this car" from "the recording differs
   from the training fleet" (European car and roads, manual driving only, 5 m/s median speed, no lead sensor).
4. Internal reference for the mechanism: on the lab's own slow human-driven test windows the LARGE recall of the size
   ensemble is 0.53-0.57 at every speed, but it ranges from 0.00 (Tesla Model S, Genesis G90) and 0.02 (Volkswagen Atlas,
   another 5.1 m^2 VW-group SUV) to 0.96 (GMC Yukon) across LARGE car models, so missing a whole LARGE model is within the
   model's internal behaviour.

Suggested use: report A2D2 as a failed single-vehicle sanity probe with the sensitivity table, outside the paper's main
results; if the owner wants a cleaner probe, a V1-style variant trained without steer_pressed (or with steer_pressed in the
dropout groups) removes the only out-of-distribution channel.

## 8. Issues met

* The GPU (16 GB) was occupied by another process during the smoke test (13.1 GB used), and `score.py --variant` died with a
  CUDA out-of-memory error inside `load_checkpoint` (`Preprocess(...).to(device)`), which the scorer's CPU fallback does not
  wrap; `--device cpu` works (34 windows take under a second). Worth wrapping `load_checkpoint` in the fallback.
* The task text suggested `lat_acc` from the IMU; the binding contract defines `lat_acc = v x yaw` (and the training tensors
  follow it exactly), so the IMU value is only a diagnostic (`--lat-acc imu`), as is `--accel imu` (the IMU a_x is rougher
  and gravity-contaminated, the speed derivative is the smoother, as the task asked to prefer).
* `steer_pressed` NaN is out of distribution (above); no adapter choice can fix it without a model variant.
* One Munich window (of 15) is below the 20 % moving rule and is dropped, as the lab rule requires.

## 9. Files

* `experiments/transfer/ds_a2d2.py` (adapter, checks, options), `experiments/transfer/DS_A2D2.md` (this note).
* `<data-root>/external/a2d2/work/`: `windows.npz` (34 x 600 x 20 float32 + channel names), `meta.parquet`,
  `windows_summary.json`, `validation.json`, `series_20180810_150607.parquet`, `series_20190401_121727.parquet`,
  `series_20190401_145936.parquet`, and the smoke-test outputs `probs__*.npz`, `score__*.md/.json` (full model, masks,
  variants).
* Raw inputs untouched: `<data-root>/external/a2d2/raw/bus/*.json` (hard links of the verified copies).
