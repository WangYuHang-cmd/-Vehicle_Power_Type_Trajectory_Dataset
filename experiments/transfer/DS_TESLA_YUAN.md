# tesla_yuan -- Yuan et al. 2024 Tesla Model 3 real-world trips (adapter notes)

Adapter: `experiments/transfer/ds_tesla_yuan.py` -> `<data-root>/external/tesla_yuan/work/{windows.npz, meta.parquet,
windows_summary.json, validation.json, trips.csv, build.log, cache_20hz/}` (built 9 Nov 2026). Window contract: `CONTRACT.md` / `common.py`.
Source: figshare article 23904102 v2 (Yuan et al., "Data and Code for Validating Generalized Energy Consumption Evaluation for
BEVs", CC BY 4.0), file 50103315 `DATA S1 update20241031.zip`; fetch record and checksums in `external/tesla_yuan/raw/FETCH.md`.
Fit grade in `CANDIDATES.md`: **sanity** (one MID BEV, speed only) -> score with the speed-only mask `V3` or a speed-only variant.

## 1. What the data is

* One physical car: 2019 Tesla Model 3 Standard Range Plus RWD (CANDIDATES 9). The authors' MATLAB code uses a test mass of
  1,919 kg and the dynamometer road-load coefficients A = 179.4 N, B = 0.28 N/(km/h), C = 0.0235 N/(km/h)^2, consistent with
  a Model 3 SR+ plus payload. The vehicle has Autopilot / TACC, but the data set and the paper do not record whether it was
  engaged on any trip.
* 106 real-world trips `data/real_world_data/001.xlsx .. 106.xlsx` (Changchun, April-May 2023), one sheet `Continuous20Hz`,
  15 columns: `Time, PhaseTime, DAActualSpeed, DAScheduleSpeed, REESSCurrent, REESSCurrent2, REESSVoltage, REESSVoltage2,
  tempAmb, elevation, lat, long, gear, tot_disc, tot_char`. 3,639,697 rows in total; trips last 714-2,974 s (11.9-49.6 min,
  median 28.0 min), 50.55 h together. Every trip starts and ends at rest.
* Used: `PhaseTime` (s, exactly 0.05-s steps = 20 Hz in all 106 trips; 0 irregular steps, 0 gaps, 0 NaN), `DAActualSpeed`
  (vehicle speed in km/h; the authors' code annotates it `% km/h`), `lat` / `long` for verification only. Cached but unused:
  `elevation`, `gear` (Tesla gear codes; 4 = D on every driving sample, 1 / 2 / 3 / 7 appear in 52 trips at stops and
  parking manoeuvres, 83 samples with gear != 4 while moving). Never read: battery current / voltage and the energy counters
  (`REESS*`, `tot_disc`, `tot_char`), which reveal the powertrain directly; `DAScheduleSpeed` (all zero on road trips);
  `tempAmb`.
* Dropped: the 46 chassis-dynamometer cycles under `data/lab_data_m1/` (CLTC, UDDS / HWY / US06 / NYCC, WLTC / CADC / JC08 and
  constant-speed runs; not road data). The 15 MATLAB scripts are treated as data and never executed.
* Speed signal character: resolution 0.07766 km/h (0.02157 m/s), exactly 0 at rest; while moving 20-35 % of consecutive
  20 Hz samples are equal, i.e. the 20 Hz grid holds / interpolates a slower CAN source (CANDIDATES: "<= 10 Hz CAN").

## 2. Time base and unit verification

* `PhaseTime` is the real clock. The GPS fix changes every 1.000 s of PhaseTime (median over the moving fix intervals of
  every one of the 106 trips; 10th-90th percentile also 1.000 s on the five trips inspected in detail), as a 1-Hz receiver
  must. The wall-clock `Time` column is a logger artefact: it
  advances 0.01 s every 3-4 rows and jumps once by about 1,650 s, so its span is 0.05-3.14 times the PhaseTime span. It is
  not used and not cached.
* `DAActualSpeed` is km/h: the GPS path length (haversine over fixes) divided by the integral of speed / 3.6 over PhaseTime
  is 1.0075-1.0153 (median 1.0127) across the 106 trips (3.6 if the column were m/s). Regressing the GPS ground speed over
  each 1-s fix interval on the CAN speed gives a slope of 1.012-1.031 (median 1.018) with intercepts of -0.03 to -0.16 m/s:
  the CAN speed reads about 1.5-2 % below the GPS ground speed, as wheel-based vehicle speeds usually do. Not corrected (the
  training `v_mps` is a CAN speed too).

## 3. Channel mapping

Order and units of `common.CHANNELS`; the scorer applies the training clip / standardisation, the adapter does not.

| channel | source | status |
|---|---|---|
| `v_mps` | `DAActualSpeed` / 3.6, 20 Hz -> 10 Hz by `resample_to_10hz(..., "decimate")`: the sample at or before each 0.1-s grid time (max age 0.075 s), never a bin mean | measured |
| `a_mps2` | `common.smooth_derivative(v_mps, hz=10, win_s=0.5)` on the 10 Hz series (central difference + 0.5-s moving average), computed inside each contiguous run | derived (see section 7) |
| `standstill` | `v_mps < 0.1` as 0 / 1; the training tensors give P(standstill \| v < 0.1) = 0.978 and P(v < 0.1 \| standstill) = 0.999, so the rule reproduces the lab flag | derived |
| `das_enabled` | constant 0; Autopilot / TACC use is not recorded -> assumed human-driven (`das_frac` = 0 everywhere) | assumed, documented |
| `steer_angle_deg`, `steer_rate_degps`, `steer_torque_norm`, `steer_pressed` | no steering measurements | missing (NaN) |
| `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2` | no IMU / yaw; GPS heading at 1 Hz is too coarse for the 10 Hz yaw convention | missing (NaN) |
| `gas_pressed`, `brake_pressed` | no pedal signals (battery current would leak the powertrain and is not a pedal) | missing (NaN) |
| `lead_valid`, `lead_d_m`, `lead_v_rel_mps`, `lead_speed_mps`, `lead_a_slope_mps2`, `thw_s`, `ttc_s` | no radar / lead | missing (NaN; `lead_valid` is NaN too, see caveat 1) |

`meta.channels_present` = `v_mps,a_mps2,standstill,das_enabled` for every window; `meta.note` repeats the derivations.

## 4. Labels

| field | value | basis |
|---|---|---|
| `dataset` / `vehicle_id` / `platform` | `tesla_yuan` / `yuan2024_model3` / `TESLA_MODEL_3` | one physical car; lab platform key of the 2017-2023 pre-facelift Model 3 |
| `model` | Tesla Model 3 | public label (row `Tesla Model 3` of `specs_external.csv`) |
| `drive_id` | `trip_001 .. trip_106` | one contiguous recording each, numbered as the public files |
| wheelbase x mean track | 2,875 mm x 1,590 mm | lab table `TESLA_MODEL_3` (tracks estimated there as 0.86 x the 1,849 mm width); Tesla's published 1,580 mm tracks give 4.543 m^2, same class |
| `footprint_m2` / `size_class` / `borderline` | 4.571 / MID / false | 0.14 m^2 above the 4.43 cut, 0.31 below 4.88 |
| `powertrain` | BEV | battery-electric car (the `REESS*` columns were not needed and not read) |
| `das_frac` | 0 | assumption of section 3 |

Tesla Model 3 is a **training** platform of `split_platform` (91 h in the lab data; the held-out Teslas are Model S and Model
Y, `release/v1/splits/splits.json`), so this set probes transfer across data source, sensor chain, region and driving style,
not unseen-model generalisation.

## 5. Window cutting and counts

`cut_windows(df10, ..., t_col="t_s", align="run")`: 600 consecutive 10 Hz samples, no gap > 0.3 s, >= 20 % of samples with
v >= 1 m/s, non-overlapping blocks restarted after every gap (there are none, so `align="drive"` gives the same blocks).

| quantity | value |
|---|---|
| trip workbooks read / dynamometer workbooks ignored | 106 / 46 |
| 20 Hz rows -> 10 Hz rows | 3,639,697 -> 1,819,879 (0 grid rows without a fresh sample) |
| source gaps / irregular steps / NaN speed | 0 / 0 / 0 |
| candidate blocks | 2,979 |
| dropped: gap inside / below the moving rule / trailing partial block | 0 / 360 (12.1 %) / 0.9 h of trip time |
| **windows** | **2,619 = 43.65 h (86.4 % of the 50.55 trip hours)** |
| vehicles / models / drives with windows | 1 / 1 / 106 (no trip without a window) |
| windows per trip | min 10, median 24, mean 24.7, max 45 |
| size_class / powertrain / borderline | MID 2,619 / BEV 2,619 / false 2,619 |

Per-trip context (`trips.csv`): durations <= 15 min 8 trips, 15-20 min 17, 20-25 min 16, 25-30 min 19, 30-40 min 32,
40-50 min 14; top speed per trip 48.7-125.6 km/h (10 trips exceed 110 km/h = expressway, 65 trips stay <= 80 km/h = urban);
share of samples at rest per trip 1-50 % (median 23 %).

## 6. Validation of the saved set (`validation.json`, "problems: none")

* Shape / dtype / channel order: `(2619, 600, 20)` float32, `windows.npz` channel list equals `common.CHANNELS`; `load_windows`
  round trip identical; `meta` has the 15 contract columns, `w_idx` consecutive inside every drive, `t0_s` increasing.
* NaN pattern: no NaN in the four declared channels (0 / 0 / 0 / 0 samples); the 16 undeclared channels are NaN in every
  sample; `channels_present` identical for all windows.
* Flags: `standstill` and `das_enabled` take only 0 / 1; `standstill` equals `v_mps < 0.1` sample by sample; mean standstill
  share 15.7 % (training mean 7.8 %); `das_enabled` = 0 everywhere.
* Physical ranges: `v_mps` 0-34.89 m/s (0-125.6 km/h; training clip -1..60), mean 9.41 m/s (33.9 km/h; training mean 17.5),
  std 7.58, median 8.88, 81.2 % of samples moving (>= 1 m/s), 15.2 % exactly zero; `a_mps2` -7.42..+3.90 m/s^2 (training clip
  +-8), std 0.696 (training 0.595), 1st / 99th percentile -1.79 / +1.98, 0.1 / 99.9th -2.85 / +2.85. 172 samples (0.011 %, 25
  windows of 22 trips) are below -4 m/s^2 and none above +4: every one was checked against the raw 20 Hz trace and is a
  genuine hard-braking event (smooth monotone descents such as 32 -> 7 km/h within one second; the whole data set has no
  single-sample speed spike and no zero drop-out while moving; the largest 20 Hz step is 0.52 m/s in trip 007).
* Window rule: `moving_frac` min 0.200, median 0.932, mean 0.812; `meta.moving_frac` equals the value recomputed from X.
* Labels: one footprint value (4.571), one class, one powertrain, `borderline` false, no mixed labels per vehicle.
* Identifiers: ids match the contract pattern; no 16-hex or route-like token in any written file (floats in the JSON / CSV are
  rounded to avoid digit runs that look like ids).
* `save_windows` warning kept on purpose: "lead_valid has NaN samples" (caveat 1).

## 7. Derived acceleration: how far from the training `aEgo`

Statistics on 600-sample blocks, same definitions for every row (cruise = v > 15 m/s and |dv over 2 s| < 0.3 m/s; roughness =
std of the sample-to-sample difference; autocorrelation after removing the block mean). Training rows: 3,000 random training
windows (`work/tensors/X_f16.npy`); Tesla rows: trips 002, 007, 030, 051, 090, 105 (148 blocks); the full Tesla set has
std 0.696.

| series | std | p1 | p99 | cruise std | roughness | lag-1 | lag-5 |
|---|---|---|---|---|---|---|---|
| training `a_mps2` (openpilot speed-KF aEgo) | 0.595 | -1.94 | +1.77 | 0.160 | 0.166 | 0.959 | 0.872 |
| training `v_mps` -> `smooth_derivative` 0.5 s (method effect alone; corr 0.85 with aEgo, rmse 0.35) | 0.665 | -1.93 | +1.76 | 0.209 | 0.144 | 0.976 | 0.784 |
| **Tesla `smooth_derivative` 0.5 s (used)** | 0.706 | -1.90 | +1.96 | 0.330 | 0.091 | 0.991 | 0.842 |
| Tesla `smooth_derivative` 1.0 s (`--accel-win 1.0`) | 0.679 | -1.82 | +1.91 | 0.299 | 0.064 | 0.995 | 0.902 |
| Tesla openpilot KF1D emulation (Q = diag(0, 100), R = 0.3, 100 Hz on the held speed) | 0.703 | -1.87 | +1.94 | 0.364 | 0.155 | 0.975 | 0.784 |

Reading: the derived series has the right overall spread and tails; its cruise variance is about twice the training value
because the source speed is a held, 0.02 m/s-quantised signal (the same method on the training speed gives 0.21), while it is
smoother in time than aEgo. Emulating openpilot's Kalman filter on the held speed does not help (its 0.08-s response amplifies
the hold steps), so the contract default stays; `--accel-win` is available for an ablation.

## 8. Caveats for the scoring step

1. **Speed-only set.** Without a mask the NaN `lead_valid` becomes the standardised mean ("unknown lead"), a state no training
   window had; `score.py --mask-channels V3` (or `lead`) sets `lead_valid := 0` and the rest to the training missing value,
   exactly as the training augmentation did. Lateral channels, pedals, torque and lead are all absent, so the honest
   comparison is a speed-only reduced-input variant (`score.py --variant`); the mask on the full model is a lower bound.
2. **`das_enabled` = 0 is an assumption.** The car has Autopilot / TACC; if it was engaged on the expressway trips, those
   windows are mislabelled as human-driven (the lab's DAS-aware training down-weights automation-engaged windows).
3. **Distribution shift in the only informative channels.** Urban Changchun stop-and-go: mean speed 9.4 m/s vs 17.5 m/s in
   training, standstill share 15.7 % vs 7.8 %, windows with 20-50 % standstill are common. A speed-only model sees mostly
   traffic, not the car.
4. **One vehicle, one region, one driver pool.** The vehicle-level result is a single decision, the vehicle bootstrap gives
   no interval; report per drive (106 trips) at most and treat the set as a BEV-positive / MID-positive sanity probe.
5. **Derived acceleration** (section 7) is noisier at cruise than the training aEgo; the hard-braking tail (down to -7.4 m/s^2)
   is real and inside the training clip.
6. **Spec provenance.** The lab's Model 3 tracks are estimates (0.86 x width); the published 1,580 mm tracks give 4.543 m^2,
   also MID and not borderline, so the label does not depend on the choice.
7. **Not usable here:** the wall-clock `Time` column (section 2); the dynamometer cycles (no road context; could serve as a
   controlled speed-profile check but are not windows of real driving); trip dates beyond "April-May 2023".
8. **Leakage guard:** battery current / voltage and energy counters are never read or cached; the per-trip cache holds only
   `t_s, v_kmh, lat, lon, elev_m, gear`.

## 9. Scorer smoke check (consumption test, not the campaign result)

`score.py --dataset tesla_yuan --mask-channels V3 --bootstrap 200` ran both default checkpoint sets on the 2,619 windows
without error (0.1 s per checkpoint on the GPU; outputs `score__*__mask-V3.md/.json`, `probs__*__mask-V3.npz` in `work/`).
With the speed-only mask on the full models the ensemble decides ICE at vehicle level for the BEV / ICE task (P(BEV) = 0.237,
window accuracy 23.7 %) and SMALL for the size task (P = 0.542 / 0.317 / 0.142 for SMALL / MID / LARGE, window accuracy
20.0 %). Both are misses; whether a speed-only variant does better is the campaign's question (caveats 1-3).

## 10. Reproduction

```
source env.sh
nice -n 10 $CPY -I experiments/transfer/ds_tesla_yuan.py                 # full build (1-3 min cold with 4 workers, ~10 s from cache_20hz/)
nice -n 10 $CPY -I experiments/transfer/ds_tesla_yuan.py --limit 3 --dry-run   # smoke test, nothing written
nice -n 10 $CPY -I experiments/transfer/ds_tesla_yuan.py --accel-win 1.0       # ablation of the acceleration smoothing
$CPY experiments/transfer/score.py --dataset tesla_yuan --mask-channels V3     # speed-only scoring
```
Options: `--raw`, `--work`, `--specs`, `--align run|drive`, `--standstill-mps 0.1`, `--workers 4`, `--no-cache`, `--dry-run`.
The adapter never writes into `raw/`; `-I` keeps the interpreter from importing anything next to the data.
