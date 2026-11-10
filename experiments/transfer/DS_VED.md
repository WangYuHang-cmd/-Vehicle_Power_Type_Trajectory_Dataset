# VED adapter (`ds_ved.py`): Vehicle Energy Dataset -> transfer windows

Written 9 Nov 2026 by the ved adapter step of the zero-shot transfer campaign. Code `experiments/transfer/ds_ved.py`
(build + validate), contract `common.py` / `CONTRACT.md`, raw data `<data-root>/external/ved/raw` (`FETCH.md`), outputs
`<data-root>/external/ved/work/` (`windows.npz`, `meta.parquet`, `windows_summary.json`, `ved_build_report.json`,
`ved_tables.md`, `ved_validation.json`, `ved_all_windows.parquet`, `ved_trips.parquet`, `ved_vehicles.parquet`, `ved_build.log`).

```
source env.sh && nice -n 10 $CPY experiments/transfer/ds_ved.py build --workers 4      # 54 weekly CSVs, ~6 min
$CPY experiments/transfer/ds_ved.py validate                                            # contract + physical-range checks
$CPY experiments/transfer/score.py --dataset ved --variant V3                            # speed-only variant (see "How to score")
```

## 1. Source

* VED (Oh, LeBlanc, Peng, IEEE T-ITS 2020; github.com/gsoh/VED, Apache-2.0): 384 privately owned cars in Ann Arbor,
  Nov 2017 - Nov 2018, OBD-II loggers; 54 weekly CSVs, 22.4 M rows, 32,552 trips, 4,848 trip-hours. The static tables give
  the powertrain per `VehId` (ICE 264, HEV 93, PHEV 24, EV 3) and a 500-lb `Generalized_Weight` bin; make / model are
  withheld, except that the README states the three EVs are 2013 Nissan Leafs (24 kWh). `Vehicle Class` is `NO DATA` for
  352 of 384 cars, `Drive Wheels` for all ICE / HEV cars.
* Dynamic columns used: `VehId`, `Trip`, `DayNum` (trip start day, only for ordering), `Timestamp(ms)` (trip-relative),
  `Vehicle Speed[km/h]`. Every engine / fuel / battery / fuel-trim / HV-battery column is direct powertrain leakage (Table II
  of the paper: engine RPM populated 100 % on ICE, 0 % on EV) and is never read.
* Timing is irregular: steps of 0.1 s (23 % of steps, bursts), 0.3-1.4 s (median 0.6 s, mean 0.78 s), tail to 2,342 s;
  0.78 % of steps exceed 3 s. Speed resolution is 1 km/h (paper, Sec. III) on every ICE car and 92 of 93 HEVs; all three
  EVs and 21 of 24 PHEVs report sub-km/h values (77-89 % non-integer samples). `(VehId, Trip)` is the trip key (`Trip`
  alone repeats across cars: 4,153 distinct values for 32,552 trips); no trip spans two weekly files.

## 2. Channel mapping

| training channel | source | status | how |
|---|---|---|---|
| `v_mps` | `Vehicle Speed[km/h]` / 3.6 | measured, **upsampled from 1 Hz** | native samples -> linear interpolation at 1 Hz nodes (integer trip seconds; a node inside a source gap > 3 s is missing) -> rounded to 1 km/h -> linear interpolation between adjacent nodes -> 10 Hz grid (`common.resample_to_10hz` / `np.interp`) |
| `a_mps2` | d`v_mps`/dt | derived | `common.smooth_derivative(v, win_s=1.0)` per gap-free run (central differences + 1-s boxcar = one source interval; training aEgo is openpilot's Kalman estimate) |
| `standstill` | `v_mps` < 0.1 m/s | derived | the training flag is 1 at 99.3 % of samples with v = 0, 84 % of 0-0.1 m/s and 1.5 % of 0.1-0.3 m/s (measured on the training tensors) |
| `das_enabled` | constant 0 | assumed | VED has no cruise / automation information; 2017-18 personal cars, so some highway cruise use is possible but unobservable |
| `steer_angle_deg`, `steer_rate_degps`, `steer_torque_norm`, `steer_pressed`, `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2` | - | missing (NaN) | no steering, IMU or GPS-heading channel (GPS is 3-s positions, not used) |
| `gas_pressed`, `brake_pressed` | - | missing (NaN) | no pedal signals |
| `lead_valid`, `lead_d_m`, `lead_v_rel_mps`, `lead_speed_mps`, `lead_a_slope_mps2`, `thw_s`, `ttc_s` | - | missing (NaN, `lead_valid` NaN = unknown) | no perception; `save_windows` warns about the NaN `lead_valid` by design |

`channels_present = v_mps,a_mps2,standstill,das_enabled` on every window. Every window note starts with
`upsampled from 1 Hz (OBD speed, irregular stamps; 1-s nodes, linear interpolation, bridge<=3s; speed rounded to 1 km/h)`
and carries `weight_bin_lb=<bin>`, `class=` / `engine=` when the static table has them, and the derivation rules.

### Pre-processing rules and why

1. **1 Hz / 1 km/h homogenisation (default).** Resolution and sampling density are class-correlated in VED: the three EVs
   and 21 / 24 PHEVs log sub-km/h speeds, ICE / HEV cars integer km/h; the EVs sample 40 % of their steps at 0.1 s
   (mean step 0.57 s) against a median of 23 % (0.78 s) for ICE cars (fewer PIDs to poll on an EV). A trace interpolated
   from the native stamps would therefore identify the plug-ins by texture (staircase ripple in `a_mps2` on ICE cars,
   none on the Leafs). Every car is brought to the same 1-Hz integer-km/h source before upsampling.
   `--native-timing` / `--native-resolution` build the untreated variant for a sensitivity run (`--dataset ved_native`).
2. **Bridging.** A node inside a source gap longer than 3 s (three nominal intervals, the analogue of the lab's 0.3 s rule
   at 10 Hz) is missing; the whole gap decides, so a long gap is never partly bridged. Missing nodes end a segment and
   `cut_windows(align="run")` restarts the 600-sample blocks after it. 174,142 source steps (0.78 %) exceeded 3 s.
3. **Glitch rule.** A node-to-node step with |dv| > 8 m/s^2 x dt (the training clip bound of `a_mps2`) is physically
   impossible. Every case inspected is the same artefact: the OBD speed frozen at 0 km/h while the car is moving, then
   jumping to 40-50 km/h within a second. The run is split at the step and the zero-speed nodes within 10 s before an upward
   step (after a downward one) are discarded as stale. 3,393 such steps in 2,495 trips (7.7 % of trips), 20,129 nodes dropped.
   Result: |a| <= 8 everywhere, max 10-Hz speed step 0.78 m/s.
4. **Smoothing window 1.0 s.** Chosen as one source interval (a 1-s boxcar nulls the 1-Hz staircase of the interpolant) and
   checked against the training tensors (3,000-window sample): autocorrelation of `a_mps2` at lags 1 / 5 / 10 / 20 samples
   0.994 / 0.883 / 0.681 / 0.525 on the VED windows against 0.961 / 0.879 / 0.767 / 0.554 for the training aEgo
   (0.5 s: 0.991 / 0.826 / 0.601 / 0.515; 1.5 s: 0.997 / 0.939 / 0.806 / 0.605). Std over moving samples 0.724 against
   0.615, 1 / 99 % quantiles -2.17 / 2.14 against -1.98 / 1.80 (1-km/h quantization plus urban stop-and-go); 10-Hz jerk std
   0.74 against 1.66 (the interpolant is smoother than the Kalman aEgo).
5. **Window rule** as in the lab (`common.cut_windows`): 600 consecutive 10-Hz samples, no gap > 0.3 s after the first
   sample, >= 20 % moving (v >= 1 m/s). 204,130 candidate blocks, 0 dropped for gaps (runs restart at gaps), 9,225
   dropped as stationary, 194,905 eligible.
6. **Cap 200 windows per vehicle**, evenly spaced over the vehicle's windows ordered by (trip day, trip, time): 244 of 384
   vehicles hit the cap; 59,563 windows kept (30.6 % of the eligible ones), 25,580 drives, 992.7 h.

## 3. Labels

| label | source | value |
|---|---|---|
| `powertrain` | static tables (`Vehicle Type` / `EngineType`) | ICE / HEV / PHEV / BEV (`EV` -> `BEV` via `common.normalise_powertrain`); task BEV vs ICE uses BEV + ICE, HEV / PHEV are scored as predicted shares only |
| `size_class`, `footprint_m2` | none for 381 cars (make / model withheld) -> NaN; the three Leafs take the Nissan Leaf ZE0 row of `specs_external.csv` (2,700 x (1,539 + 1,534) / 2 mm = 4.149 m^2, SMALL, not borderline; DOE AVT fact sheet) | NaN / SMALL (431 windows, 3 vehicles); `--no-leaf-footprint` reverts to NaN everywhere |
| `model` | the only public grouping is powertrain x weight bin | `VED <powertrain> <weight> lb` (e.g. `VED ICE 3500 lb`), `VED ICE weight n/a` (16 cars without a weight), `Nissan Leaf (2013)`; the scorer's per-model table is therefore the weight-bin table (the "weight-bin ordinal check" of CANDIDATES.md) |
| `vehicle_id` | `VehId` | 384 physical cars (bootstrap clusters) |
| `drive_id` | `<VehId>-<Trip>` | one trip = one contiguous recording; `Trip` alone is not unique, hence the prefix |
| `platform` | - | None |
| `note` | - | weight bin, vehicle class / engine configuration when present, derivation rules, `upsampled from 1 Hz` |

## 4. Counts

Per powertrain (`ved_tables.md`; dense step = source step <= 0.25 s, means over vehicles):

| powertrain | vehicles | trips | trip h | eligible windows | kept windows | kept h | kept drives | at cap | dense-step share | non-integer speed share | mean step s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ICE | 264 | 18,936 | 2,742.0 | 123,073 | 38,828 | 647.1 | 15,453 | 153 | 0.227 | 0.000 | 0.78 |
| HEV | 93 | 9,507 | 1,411.7 | 45,377 | 16,206 | 270.1 | 7,456 | 70 | 0.214 | 0.004 | 0.91 |
| PHEV | 24 | 3,605 | 614.9 | 23,896 | 4,098 | 68.3 | 2,354 | 19 | 0.204 | 0.785 | 0.70 |
| BEV (2013 Leaf) | 3 | 504 | 79.8 | 2,559 | 431 | 7.2 | 317 | 2 | 0.396 | 0.893 | 0.57 |
| total | 384 | 32,552 | 4,848.4 | 194,905 | 59,563 | 992.7 | 25,580 | 244 | | | |

Per model (= powertrain x weight bin):

| model | vehicles | kept windows | kept drives | kept h |
|---|---|---|---|---|
| VED ICE 2500 lb | 36 | 5,534 | 2,400 | 92.2 |
| VED ICE 3000 lb | 53 | 7,457 | 3,259 | 124.3 |
| VED ICE 3500 lb | 82 | 13,279 | 5,307 | 221.3 |
| VED ICE 4000 lb | 29 | 4,256 | 1,557 | 70.9 |
| VED ICE 4500 lb | 35 | 4,611 | 1,898 | 76.8 |
| VED ICE 5000 lb | 5 | 577 | 154 | 9.6 |
| VED ICE 5500 lb | 4 | 331 | 109 | 5.5 |
| VED ICE 6000 lb | 3 | 198 | 36 | 3.3 |
| VED ICE 6500 lb | 1 | 15 | 5 | 0.3 |
| VED ICE weight n/a | 16 | 2,570 | 728 | 42.8 |
| VED HEV 2000 lb | 1 | 164 | 44 | 2.7 |
| VED HEV 2500 lb | 6 | 872 | 455 | 14.5 |
| VED HEV 3000 lb | 52 | 9,485 | 4,304 | 158.1 |
| VED HEV 3500 lb | 26 | 4,423 | 2,005 | 73.7 |
| VED HEV 4000 lb | 5 | 662 | 382 | 11.0 |
| VED HEV 4500 lb | 2 | 400 | 146 | 6.7 |
| VED HEV 5000 lb | 1 | 200 | 120 | 3.3 |
| VED PHEV 3000 lb | 2 | 400 | 329 | 6.7 |
| VED PHEV 3500 lb | 3 | 374 | 166 | 6.2 |
| VED PHEV 4000 lb | 19 | 3,324 | 1,859 | 55.4 |
| Nissan Leaf (2013) | 3 | 431 | 317 | 7.2 |

Per weight bin, all powertrains (vehicles / kept windows): 2000 lb 1 / 164; 2500 lb 42 / 6,406; 3000 lb 107 / 17,342;
3500 lb 114 / 18,507; 4000 lb 53 / 8,242; 4500 lb 37 / 5,011; 5000 lb 6 / 777; 5500 lb 4 / 331; 6000 lb 3 / 198;
6500 lb 1 / 15; no weight 16 / 2,570.

## 5. Validation (`ds_ved.py validate`, 0 problems; `ved_validation.json`)

* Shape (59,563, 600, 20) float32; `channels_present` identical on every window; no NaN in the four present channels,
  NaN everywhere in the 16 absent ones; flags 0 / 1; `das_enabled` = 0 and `das_frac` = 0; `standstill == (v < 0.1 m/s)`.
* Speed 0-42.8 m/s (max 154 km/h; the raw library tops at 173 km/h, no 255-km/h OBD error values), mean 12.2 m/s, median
  12.9, 99 % 32.5 (training median 17.6, 99 % 36.3: Ann Arbor urban driving); moving share 0.886 (training 0.915),
  standstill share 0.092 (training 0.074); every window has moving_frac >= 0.2 and `meta.moving_frac` matches X.
* |a| <= 8 m/s^2 everywhere, |a| > 5 in 0.058 % of samples; quantiles 0.1 / 1 / 50 / 99 / 99.9 % = -3.44 / -2.11 / 0.00 /
  2.11 / 4.26; largest 10-Hz speed step 0.78 m/s (bound 0.80 from the glitch rule).
* <= 200 windows per vehicle; one model / powertrain / size_class per vehicle; `drive_id` = `<VehId>-<Trip>` with the
  vehicle prefix; `(drive_id, w_idx)` unique; every note carries `upsampled from 1 Hz`; powertrain vocabulary
  {ICE, HEV, PHEV, BEV}; Leaf windows SMALL / 4.149 / not borderline; no 16-hex identifier-like token in any meta string.
* Per powertrain inside the kept windows: mean speed ICE 11.5, HEV 13.8, PHEV 12.4, BEV 12.0 m/s; moving share 0.85 /
  0.97 / 0.91 / 0.92; std of `a_mps2` over moving samples 0.74 / 0.69 / 0.70 / 0.76.

## 6. How to score

* The matching reduced-input model is the speed-only variant **V3** (`v_mps, a_mps2, standstill`): `score.py --dataset ved
  --variant V3`. The full 20-channel model may be run with `--mask-channels V3` only as a secondary number: the missing
  lateral channels are outside the training augmentation (CONTRACT.md), and `das_enabled` = 0 is an assumption.
* Powertrain: 264 ICE vehicles against 3 BEV vehicles (the Leafs), so the BEV side is a sanity check (expected-label hit
  rate per vehicle / drive), not an accuracy with an interval; the 93 HEV and 24 PHEV cars are reported as predicted shares
  (VARIANTS.md: V3 BEV-vs-ICE reference about 88 drive / 91 vehicle, BEV recall 78). Compare with the lab's human-only
  (`das_enabled` = 0) windows.
* Size: only the three Leafs carry a label (SMALL); the per-model (= weight bin) predicted shares are the ordinal check:
  the SMALL share should fall and the LARGE share rise with the weight bin. V3 size numbers are indicative only
  (VARIANTS.md: 45-46 drive / vehicle reference against chance 33).

## 7. Caveats

1. Speed is the only measured channel and it is upsampled from an irregular ~1 Hz OBD stream at 1 km/h; `a_mps2` is a
   smoothed derivative of that interpolant (wider tails, less 10-Hz texture than the training aEgo), `standstill` a speed
   threshold, `das_enabled` an assumed 0. Lateral channels, pedals and lead are missing -> V3 only.
2. Class-correlated logging artefacts (resolution, sampling density, Table II population pattern) were homogenised, but the
   EV loggers also differ in ways that cannot be removed (0.1-s bursts more frequent, shorter mean step); the
   `ved_native` sensitivity build shows how much the homogenisation matters.
3. Powertrain classes are heavily imbalanced (3 BEV vehicles) and the "models" are weight bins, not car models: the
   scorer's vehicle-level bootstrap is meaningful for ICE (264 vehicles) only.
4. Trips are trip-relative in time (`t0_s` counts from the trip start; the date is in `ved_all_windows.parquet.DayNum`,
   not in meta); start / end positions were fogged by the authors, irrelevant here.
5. All 1,396 NaN speed rows sit in the first two weekly files (ignored); 16 ICE cars have no weight bin (`VED ICE weight n/a`).
6. `common.resample_to_10hz(kind="linear", max_age_s=...)` bridges the first `max_age_s` seconds of a longer gap (a ramp
   towards the far sample) because its age test looks only at the previous sample; this adapter therefore computes node
   validity from the length of the enclosing source interval (`node_values`). Other adapters that interpolate sparse
   sources should do the same or ask for a whole-gap option in `common.py`.
7. Deviations from the task text: `drive_id` is `<VehId>-<Trip>` (not `Trip`, which is not unique) and the three Leafs carry
   the public ZE0 footprint (SMALL) instead of NaN because their make / model is public; `--no-leaf-footprint` disables it.
