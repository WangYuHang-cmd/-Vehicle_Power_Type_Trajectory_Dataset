# OpenACC adapter: `ds_openacc.py` -> window sets `openacc` and `openacc_500x`

Adapter for the EC JRC OpenACC database (CC BY 4.0; raw copy under `external/openacc/raw/`, see its `FETCH.md` and
`external/MANIFEST.md` 2.2) in the zero-shot transfer campaign (`CANDIDATES.md` 2.4, contract `CONTRACT.md` /
`common.py`). Built 9 Nov 2026 with the default options; everything below is reproduced by

```
source env.sh && $CPY -I experiments/transfer/ds_openacc.py          # both sets, ~2 min
```

Two window sets, because the one-vehicle campaign carries no lead information and should be scored with the speed-only
mask while the platoon campaigns carry the whole longitudinal group:

| set | content | windows | vehicles | drives | models | hours | score with |
|---|---|---|---|---|---|---|---|
| `openacc` (`external/openacc/work/`) | six platoon campaigns (AstaZero, Casale, Cherasco, JRC low speed, Vicolungo, ZalaZone), every named non-hybrid car plus the two mild hybrids | 4,540 | 21 | 630 | 20 | 75.7 | `--mask-channels V2` or the longitudinal-only variant |
| `openacc_500x` (`external/openacc_500x/work/`) | one Fiat 500X diesel, 20 volunteer drivers, 438 trips, manual driving (CAN/OBD speed only) | 13,523 | 1 | 429 | 1 | 225.4 | `--mask-channels V3` or the speed-only variant |

Each `work/` folder holds `windows.npz` + `meta.parquet` + `windows_summary.json` (the contract), and the adapter's QC
files `qc_vehicle_files.csv` (one row per vehicle x run file), `qc_summary.json`, `qc_tables.md` (the generated tables
quoted below) and `build.log`. The raw folder is read only.

## 1. Channel mapping

Platoon CSV: 5 metadata rows (`Date`, `Vehicle_order`, `Number_of_vehicles`, `ACC` flag 0 / 1 / 2 / blank,
`Distance_setting`), then the header; column suffix n = n-th vehicle of `Vehicle_order`, `IVSn` = bumper-to-bumper
spacing between vehicles n and n+1, `Drivern` = `Human` / `ACC` per sample (absent in ZalaZone and in several JRC low
speed vehicles). One-vehicle CSV: `Time, Speed, Engine Speed, Gear` at 10 Hz (`Engine Speed` and `Gear` are never read).

| training channel | `openacc` (platoon campaigns) | status | `openacc_500x` | status |
|---|---|---|---|---|
| `v_mps` | `Speedn` (GNSS Doppler speed, m/s; AstaZero: RT-Range dGNSS 100 Hz downsampled; Cherasco / Vicolungo / Casale / JRC low speed: u-blox 3-5 Hz cubic-splined to 10 Hz; ZalaZone: VBOX + u-blox fused, B-spline) | measured (GNSS, not CAN) | `Speed` (CAN/OBD, interpolated to 10 Hz by JRC) | measured |
| `a_mps2` | `common.smooth_derivative(v)` per segment: central difference, 0.5-s moving average | derived | same | derived |
| `standstill` | `v < 0.2 m/s` (training: CAN flag; 99.9 % of training standstill samples have v < 0.10 m/s, 94 % of training samples below 0.2 m/s are standstill) | derived | same | derived |
| `das_enabled` | `Drivern == ACC` per sample where the column exists; else campaign `ACC` flag 1 / 0; else protocol default (follower 1, leader 0) -- see section 2 | measured (per sample) / protocol | 0 (manual driving, no ACC) | protocol |
| `lead_valid` | 1 where `IVS(n-1)` and `Speed(n-1)` are finite and 0.5 m < IVS <= 150 m; 0 for leaders, for vehicles whose predecessor dropped out of the file, and on the cleaned spacing samples | measured | 0 (no lead sensor) | masked group |
| `lead_d_m` | `IVS(n-1)` (dGNSS / GNSS bumper-to-bumper spacing) | measured | NaN | - |
| `lead_v_rel_mps` | `Speed(n-1) - Speedn` (lead minus ego, the training sign) | measured | NaN | - |
| `lead_speed_mps` | `v + lead_v_rel` (= `Speed(n-1)`) | measured | NaN | - |
| `lead_a_slope_mps2` | `common.lead_channels`: causal 1-s least-squares slope of the lead speed inside lead episodes, >= 3 samples, clipped +-4 | derived (lab rule) | NaN | - |
| `thw_s`, `ttc_s` | `common.lead_channels`: d / v (v > 0.5 m/s, d > 0.5 m); d / closing speed (closing > 0.1 m/s) | derived (lab rule) | NaN | - |
| `steer_angle_deg`, `steer_rate_degps`, `steer_torque_norm`, `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2`, `steer_pressed` | not in the data (curvature from GNSS heading would be usable only at AstaZero and is left missing everywhere, as CANDIDATES.md 2.4 asks) | missing (outside the training augmentation -> V2 mask / variant) | missing | missing |
| `gas_pressed`, `brake_pressed` | not in the data | missing (pedals group) | missing | missing |

Channels present in `openacc`: `v_mps, a_mps2, standstill, das_enabled, lead_valid, lead_d_m, lead_v_rel_mps,
lead_speed_mps, lead_a_slope_mps2, thw_s, ttc_s`; in `openacc_500x`: `v_mps, a_mps2, standstill, das_enabled, lead_valid`.
The adapter never clips, standardises or fills (the scorer's `Preprocess` does).

## 2. `das_enabled` rule and sources

Priority per vehicle and file: (1) the per-sample `Drivern` column (`ACC` -> 1, `Human` -> 0; AstaZero, Cherasco,
Vicolungo, Casale follower, part of JRC low speed); (2) a leader without a Driver column is taken as human-driven (the
campaign flag describes the followers: Vicolungo `ACC = 1` files carry `Driver1 = Human`; affects the Casale Rexton and
two JRC low speed S-Max runs, 75 windows); (3) the campaign `ACC` flag 1 -> 1, 0 -> 0 (ZalaZone, where no Driver column
exists; four handling runs are manual); (4) `ACC = 2` (mixed) or blank with no Driver column -> follower 1 by protocol
("followers drove with ACC on apart from manual-braking instances", 19 windows).

| das source | vehicle-files | windows |
|---|---|---|
| Driver column (per sample) | 118 | 1,954 |
| campaign ACC flag = 0 | 44 | 247 |
| campaign ACC flag = 1 | 460 | 2,245 |
| leader without Driver column: assumed human | 11 | 75 |
| protocol default (ACC flag 2 or blank, follower on ACC) | 3 | 19 |

DAS share per campaign (window mean): AstaZero 0.83 (two manual laps, leader on ACC), Casale 0.00 (Rexton leader),
Cherasco 0.36, JRC low speed 0.91, Vicolungo 0.45, ZalaZone 0.90; the whole set 0.78 (training: 0.56). Compare against
the lab's automation-engaged windows, not the human-only protocol (CONTRACT.md pitfall 5). The S-Max leading the JRC low
speed runs 18-24 shows `Driver1 = ACC` (cruise control without a target): two neighbouring runs of the same session
without a Driver column fall under rule (2) and are labelled human (about 5 windows).

## 3. Lead group: spacing QC

`lead_valid` requires a finite spacing and predecessor speed and 0.5 m < IVS <= 150 m. The 0.5 m floor is the lab's
`LEAD_VALID_DREL_MIN_M`; the 150 m ceiling emulates the radar range of the training leads (99.9 % of the training
`lead_d_m` < 135 m, maximum 165 m), so motorway platoon gaps of 150-450 m (Vicolungo, ZalaZone handling) become "no lead"
exactly as they would on a radar car. Negative spacings (the files listed in the campaign notes; GNSS error when the cars
are very close) are set to "no lead" on those samples, never clipped or shifted.

| spacing QC (platoon set, samples) | total | by campaign |
|---|---|---|
| valid lead rows (before windowing) | 2,195,634 | - |
| IVS < 0 (negative spacing) | 2,871 | Cherasco 869, Vicolungo 4, ZalaZone 1,998 |
| 0 <= IVS <= 0.5 m | 2,922 | Cherasco 493, JRC low speed 24, Vicolungo 67, ZalaZone 2,338 |
| IVS > 150 m | 2,833 | Vicolungo 1,701, ZalaZone 1,132 |
| IVS NaN while the follower moves | 35,365 | JRC low speed 7,797, ZalaZone 27,568 |

Lead share inside the windows: 0.75 (training 0.54); `lead_a_slope` finite on 99.9 % and `thw` on 97.4 % of the lead
samples (training 86 % / 91 %: GNSS spacing has no radar dropouts). Leaders (Audi A8, Fiat 500X, VW Polo, Space Star,
Rexton, S-Max in JRC low speed runs 18-24) have `lead_valid = 0` throughout (their `lead_frac` is 0 in the per-model
table), as a radar car with nothing in range would.

## 4. Speed QC (`clean_speed`)

Applied to every speed series before the derivative and the window cut; counts are samples on the 10 Hz grid.

1. Speed outside [-1, 70] m/s -> missing (0 samples hit).
2. **Frozen sensor**: >= 30 identical consecutive finite values (3.0 s; identical = within 1e-5 m/s) while v > 0.5 m/s ->
   missing. `openacc_500x`: 128,672 samples (3.57 h) in 8 trips -- one trip holds 63,995 samples (1.8 h) of constant
   25.8 m/s, two trips (D5 parts 7 and 8) are frozen end to end and yield no windows, three more trips lose 15k / 5.7k /
   0.9k samples. `openacc`: 414 samples in Cherasco (120) and Vicolungo (294): a stale u-blox speed carried through the
   JRC cubic spline for 3-4 s (values repeat to 1e-9). The 1-2 s plateaus of the AstaZero speeds (quantised to 0.01 m/s,
   cars on ACC) are below the threshold and kept (an earlier 1.0-s threshold removed 956 legitimate AstaZero samples).
3. **Impossible step**: |dv| / dt > 15 m/s^2 between finite samples at most 0.3 s apart (1.5 m/s per 0.1 s; no car does
   it) -> the series is split there (no window straddles the jump) and a hold of >= 2 identical values ending at the jump
   (the stale samples before it) -> missing. `openacc_500x`: 14 steps in 11 trips, 33 hold samples (hold-then-jump
   pattern of the OBD interpolation, e.g. 11.2 m/s held for 0.9 s then 24.8 m/s). `openacc`: 1 step (Vicolungo Space
   Star, a 3-sample GNSS hold followed by +2.7 m/s), 3 hold samples.
4. Missing speed samples are dropped from the 10 Hz frame (11,153 raw NaN rows: JRC low speed 668, ZalaZone 10,485);
   `common.cut_windows` then applies the lab rules: a gap is > 0.3 s between consecutive samples (holes of 1-2 missing
   samples are tolerated, as in the lab), 600 consecutive samples, >= 20 % moving (v >= 1 m/s), `align='run'` (blocks
   restart after every gap or step). The derivative bridges tolerated holes linearly for the differentiation only.

Acceleration excursions that remain: `openacc` has 13 samples with |a| > 8 m/s^2 in 3 windows (2 above 10 m/s^2:
spline overshoot at hard accelerations of the Mini Cooper and the S-Max in Vicolungo, 25 -> 33 m/s in 0.8 s); the
scorer's `Preprocess` clips `a_mps2` to +-8. `openacc_500x` has 43 such samples in 6 windows (hardest -9.3 m/s^2, no
step > 1.5 m/s per sample left).

## 5. Identifiers and windows

* `vehicle_id` = `<campaign>/<model>` (one physical car per model per campaign), except the JRC fleet cars that the
  spec files show as the same vehicle in two campaigns, merged under one id: `JRC fleet/Ford S-Max` (Vicolungo 2019 and
  JRC low speed 2020, both 110 kW diesel MY2018) and `JRC fleet/Fiat 500X` (one-vehicle campaign 2016-17 and Cherasco
  leader 2018, both 103 kW diesel MY2016; the `openacc_500x` set uses the same id). The Tesla Model 3 of AstaZero
  (150 kW in the spec file) and of ZalaZone (250 kW) are different cars and keep two ids (`--no-merge-fleet` keeps every
  campaign separate).
* `drive_id` = `<campaign>/<run file>/<header name>` (one contiguous recording of one vehicle; the scorer's drive level);
  `openacc_500x`: `OneVehicle/D<driver>_part<n>` (one trip; the driver id is also in `note`).
* `platform` = the lab platform name only where `specs_external.csv` copied the lab table verbatim (Tesla Model 3 / S / X,
  VW e-Golf = VOLKSWAGEN_GOLF_MK7); otherwise None.
* `note` per window: campaign and acquisition, platoon position and role, das source, lead source, derivation notes, the
  JRC spec-file propulsion / power / model year, and the MHEV or fleet-merge remarks. No device or route identifiers
  exist in this dataset; the HEX12 / HEX16 scan of ids, notes and this file is clean.

## 6. Labels

Header name -> canonical public model -> `specs_external.csv` row (footprint = wheelbase x mean track, locked cuts
4.43 / 4.88 m^2, borderline = |footprint - cut| < 0.05 or conflicting published dimensions) cross-checked against the
campaign vehicle-specification CSV (propulsion type, power, model year). Decisions follow CANDIDATES.md 2.3-2.4 and the
task brief: the Peugeot (3008 in every header, 5008 in the spec file) and the unnamed BMW / Smart / Skoda leaders are
excluded; hybrids and the fuel-cell car are excluded by default (`--keep-hybrids` keeps them with HEV / PHEV / FCEV
labels, usable for the size task); the two 48 V / 12 V mild hybrids (GLE 450, A4 Avant) are kept for the size task and
labelled `HEV` as in the JRC spec file so that the scorer reports them as predicted shares only and keeps them out of the
BEV / ICE score (`--mhev-as-ice` applies the lab's mild-hybrid = ICE convention instead, `--drop-mhev` removes them).

| campaign | header name | model | JRC spec (propulsion, kW, MY) | footprint m^2 | size | powertrain | borderline | decision | windows |
|---|---|---|---|---|---|---|---|---|---|
| AstaZero | Audi(A8) | Audi A8 | ICE 210 2018 | 4.912 | LARGE | ICE | yes (0.03 above the MID/LARGE cut) | leader, included | 220 |
| AstaZero | Tesla(Model3) | Tesla Model 3 | BEV 150 2019 | 4.571 | MID | BEV | no | included | 220 |
| AstaZero | BMW(X5) | BMW X5 | ICE 195 2018 | 4.985 | LARGE | ICE | no | included | 220 |
| AstaZero | Audi(A6) | Audi A6 | ICE 150 2018 | 4.747 | MID | ICE | no | included | 220 |
| AstaZero | Mercedes(AClass) | Mercedes-Benz A-Class | ICE 165 2019 | 4.249 | SMALL | ICE | no | included | 220 |
| Casale | Rexton | SsangYong Rexton | ICE 133.1 2019 | 4.699 | MID | ICE | no | leader, included | 69 |
| Casale | Hyundai | Hyundai Nexo | Fuel Cell Electric 120 2019 | 4.517 | MID | FCEV | no | excluded (FCEV; `--keep-hybrids`) | 0 |
| Cherasco | Fiat(500X) | Fiat 500X | ICE 103 2016 | 3.971 | SMALL | ICE | no | leader, included (JRC fleet id) | 182 |
| Cherasco | Hyundai(Ioniqhybrid) | Hyundai Ioniq Hybrid | HEV 104 2018 | 4.239 | SMALL | HEV | no | excluded (HEV) | 0 |
| Cherasco | Volvo(XC40) | Volvo XC40 | ICE 140 2018 | 4.360 | SMALL | ICE | no | included | 184 |
| Cherasco | VW(Polo) | Volkswagen Polo (Mk5 6R, 2010) | ICE 63 2010 (petrol/LPG) | 3.605 | SMALL | ICE | no | leader, included | 82 |
| JRC low speed | BMW | - | not in the spec file | - | - | - | - | excluded (unnamed) | 0 |
| JRC low speed | Hyundai | Hyundai Nexo | Fuel Cell Electric 120 2019 | 4.517 | MID | FCEV | no | excluded (FCEV) | 0 |
| JRC low speed | Ford | Ford S-Max | ICE 110 2018 | 4.575 | MID | ICE | no | included (JRC fleet id; leader in runs 18-24) | 103 |
| Vicolungo | Mitsubishi(SpaceStar) | Mitsubishi Space Star | ICE 59 2018 | 3.485 | SMALL | ICE | no | leader, included | 114 |
| Vicolungo | Ford(S-Max) | Ford S-Max | ICE 110 2018 | 4.575 | MID | ICE | no | included (JRC fleet id) | 114 |
| Vicolungo | Peugeot(3008GTLine) | - | ICE 130 2018 (spec file: 5008 GT Line) | 4.265 / 4.538 | SMALL / MID | ICE | - | excluded (identity unresolved) | 0 |
| Vicolungo | KIA(Niro) | Kia Niro PHEV | PHEV 77.2 2019 | 4.293 | SMALL | PHEV | no | excluded (PHEV) | 0 |
| Vicolungo | Mini(Cooper) | MINI Cooper | ICE 100 2018 | 3.745 | SMALL | ICE | no | included | 42 |
| Vicolungo | VW(GolfE) | Volkswagen e-Golf | BEV 100 2018 | 4.020 | SMALL | BEV | no | included | 73 |
| Vicolungo | Mitsubishi(OutlanderPHEV) | Mitsubishi Outlander PHEV | PHEV 99 2018 | 4.112 | SMALL | PHEV | no | excluded (PHEV) | 0 |
| ZalaZone | SMART_TARGET / SKODA_TARGET | - | not in the spec file | - | - | - | - | excluded (unnamed leaders) | 0 |
| ZalaZone | BMW_I3 | BMW i3s REx | HEV 135 2018 | 4.073 | SMALL | PHEV (range extender) | no | excluded (PHEV) | 0 |
| ZalaZone | MERCEDES_GLE450 | Mercedes-Benz GLE 450 4MATIC | HEV 270 2019 (48 V EQ Boost) | 5.023 | LARGE | HEV (MHEV; ICE by the lab convention) | no | included for size; outside the BEV/ICE score | 360 |
| ZalaZone | JAGUAR_I_PACE | Jaguar I-Pace | BEV 294 2019 | 4.917 | LARGE | BEV | yes (4.89-4.94 by source) | included | 292 |
| ZalaZone | TESLA_MODELX | Tesla Model X | BEV 386 2016 | 5.061 | LARGE | BEV | no | included | 345 |
| ZalaZone | TESLA_MODEL3 | Tesla Model 3 | BEV 250 2019 | 4.571 | MID | BEV | no | included (second car) | 266 |
| ZalaZone | TESLA_MODELS | Tesla Model S | BEV 244 2018 | 4.973 | LARGE | BEV | no | included | 334 |
| ZalaZone | AUDI_E_TRON | Audi e-tron | BEV 300 2019 | 4.841 | MID | BEV | yes (0.04 below the LARGE cut) | included | 350 |
| ZalaZone | TOYOTA_RAV4 | Toyota RAV4 Hybrid | HEV 115 2019 | 4.344 | SMALL | HEV | no | excluded (HEV) | 0 |
| ZalaZone | MAZDA_3 | Mazda 3 (BP) | ICE 96 2019 (24 V M Hybrid = ICE) | 4.292 | SMALL | ICE | no | included | 305 |
| ZalaZone | AUDI_A4 | Audi A4 Avant (B9) | HEV 140 2019 (12 V MHEV) | 4.409 | SMALL | HEV (MHEV; ICE by the lab convention) | yes (0.02 below the SMALL/MID cut) | included for size; outside the BEV/ICE score | 225 |
| OneVehicle | Fiat500X | Fiat 500X | ICE 103 2016 | 3.971 | SMALL | ICE | no | `openacc_500x`, speed-only | 13,523 |

Size-task note: the Audi A8 (4.912), Audi e-tron (4.841), Jaguar I-Pace (4.917) and Audi A4 Avant (4.409) carry
`borderline = True`; the scorer repeats the size metrics without them. Powertrain task in `openacc`: 1,880 BEV windows
(7 cars / 6 models) vs 2,075 ICE windows (12 cars / 12 models); 585 HEV-labelled windows (GLE 450, A4 Avant) are
reported as predicted shares only.

## 7. Counts

### `openacc` per model (generated, `qc_tables.md`)

| model | windows | vehicles | drives | hours | footprint_m2 | size_class | powertrain | borderline | das_frac | lead_frac | v_mean_mps | v_max_mps | moving_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Audi A4 Avant | 225 | 1 | 47 | 3.75 | 4.409 | SMALL | HEV | True | 0.924 | 0.989 | 9.5 | 26.8 | 0.931 |
| Audi A6 | 220 | 1 | 10 | 3.67 | 4.747 | MID | ICE | False | 0.791 | 1.000 | 18.5 | 33.3 | 0.989 |
| Audi A8 | 220 | 1 | 10 | 3.67 | 4.912 | LARGE | ICE | True | 1.000 | 0.000 | 18.5 | 27.7 | 0.991 |
| Audi e-tron | 350 | 1 | 67 | 5.83 | 4.841 | MID | BEV | True | 0.903 | 0.777 | 9.9 | 24.2 | 0.959 |
| BMW X5 | 220 | 1 | 10 | 3.67 | 4.985 | LARGE | ICE | False | 0.791 | 1.000 | 18.5 | 32.7 | 0.990 |
| Fiat 500X | 182 | 1 | 6 | 3.03 | 3.971 | SMALL | ICE | False | 0.000 | 0.000 | 28.8 | 45.8 | 0.976 |
| Ford S-Max | 217 | 1 | 36 | 3.62 | 4.575 | MID | ICE | False | 0.831 | 0.789 | 19.0 | 40.8 | 0.983 |
| Jaguar I-Pace | 292 | 1 | 60 | 4.87 | 4.917 | LARGE | BEV | True | 0.887 | 0.789 | 9.7 | 21.3 | 0.963 |
| MINI Cooper | 42 | 1 | 5 | 0.70 | 3.745 | SMALL | ICE | False | 0.000 | 1.000 | 29.8 | 40.2 | 1.000 |
| Mazda 3 | 305 | 1 | 63 | 5.08 | 4.292 | SMALL | ICE | False | 0.889 | 0.298 | 9.9 | 25.6 | 0.953 |
| Mercedes-Benz A-Class | 220 | 1 | 10 | 3.67 | 4.249 | SMALL | ICE | False | 0.791 | 1.000 | 18.5 | 34.4 | 0.989 |
| Mercedes-Benz GLE 450 4MATIC | 360 | 1 | 71 | 6.00 | 5.023 | LARGE | HEV | False | 0.911 | 0.995 | 10.0 | 24.1 | 0.965 |
| Mitsubishi Space Star | 114 | 1 | 12 | 1.90 | 3.485 | SMALL | ICE | False | 0.000 | 0.000 | 27.8 | 39.4 | 0.978 |
| SsangYong Rexton | 69 | 1 | 9 | 1.15 | 4.699 | MID | ICE | False | 0.000 | 0.000 | 32.3 | 40.1 | 0.988 |
| Tesla Model 3 | 486 | 2 | 58 | 8.10 | 4.571 | MID | BEV | False | 0.844 | 0.996 | 13.5 | 34.5 | 0.977 |
| Tesla Model S | 334 | 1 | 69 | 5.57 | 4.973 | LARGE | BEV | False | 0.898 | 0.991 | 9.8 | 27.5 | 0.957 |
| Tesla Model X | 345 | 1 | 69 | 5.75 | 5.061 | LARGE | BEV | False | 0.904 | 0.799 | 9.9 | 25.2 | 0.965 |
| Volkswagen Polo | 82 | 1 | 5 | 1.37 | 3.605 | SMALL | ICE | False | 0.000 | 0.000 | 28.0 | 33.9 | 0.982 |
| Volkswagen e-Golf | 73 | 1 | 7 | 1.22 | 4.020 | SMALL | BEV | False | 0.901 | 1.000 | 27.0 | 40.3 | 0.974 |
| Volvo XC40 | 184 | 1 | 6 | 3.07 | 4.360 | SMALL | ICE | False | 0.865 | 0.995 | 28.5 | 46.3 | 0.967 |

The Mazda 3's low `lead_frac` (0.30): in 42 of its 64 ZalaZone series the vehicle ahead is absent from the file (the
Skoda target car, whose columns are missing in 42 of the 73 files, or a RAV4 that dropped out), so no spacing exists and
`lead_valid` is 0 by construction, as for a radar car with nothing in range.

### `openacc` per campaign and per class

| campaign | windows | vehicles | models | drives | hours | das_frac |
|---|---|---|---|---|---|---|
| AstaZero | 1,100 | 5 | 5 | 50 | 18.3 | 0.833 |
| Casale | 69 | 1 | 1 | 9 | 1.15 | 0.000 |
| Cherasco | 448 | 3 | 3 | 17 | 7.47 | 0.355 |
| JRC low speed | 103 | 1 | 1 | 24 | 1.72 | 0.908 |
| Vicolungo | 343 | 4 | 4 | 36 | 5.72 | 0.445 |
| ZalaZone | 2,477 | 8 | 8 | 494 | 41.3 | 0.900 |

| size_class | windows | vehicles | models | | powertrain | windows | vehicles | models |
|---|---|---|---|---|---|---|---|---|
| SMALL | 1,427 | 9 | 9 | | BEV | 1,880 | 7 | 6 |
| MID | 1,342 | 6 | 5 | | ICE | 2,075 | 12 | 12 |
| LARGE | 1,771 | 6 | 6 | | HEV (two MHEVs, shares only) | 585 | 2 | 2 |

Window budget: 636 vehicle-file series read, 2,950,320 rows, 4,564 candidate blocks, 24 dropped by the moving rule, 0
by gaps (the files are gap-free; the dropped NaN rows only shorten runs), 4,540 windows in 630 series. Skipped series:
95 ZalaZone vehicle columns absent (drop-outs), 159 hybrid / FCEV, 42 unnamed leaders, 5 Peugeot.

### `openacc_500x`

13,523 windows (225.4 h) from 429 of the 438 trips (nine trips yield none: two frozen end to end, the others too short
or stationary after cleaning); 19,718 candidate blocks, 6,195 dropped by the moving rule (idling at trip ends; the
campaign's 336 h include 7 % standstill in the kept windows and much more in the dropped blocks). Windows per driver:
D1 419, D2 539, D3 2,076, D4 243, D5 746, D6 107, D7 162, D8 685, D9 381, D10 305, D11 810, D12 234, D13 1,205,
D14 1,031, D15 427, D16 1,986, D17 395, D18 362, D19 339, D20 1,071. Speed mean 16.8 m/s, maximum 48.9 m/s
(176 km/h, within the car's 190 km/h top speed), moving share 0.92, standstill share 0.067, `das_enabled` 0 throughout.

## 8. Validation (adapter checks, both sets pass unless stated)

* Shapes and contract: `common.validate_windows` passes with no warnings for both sets (flags 0 / 1 only, no window
  below the moving rule, `channels_present` consistent, labels unique per vehicle).
* No NaN inside the declared channels: `v_mps`, `a_mps2`, `standstill`, `das_enabled`, `lead_valid` are finite in every
  sample; `lead_d_m`, `lead_v_rel_mps`, `lead_speed_mps` are finite exactly where `lead_valid = 1` (and NaN where it is
  0); `lead_speed_mps = v + lead_v_rel_mps` to 1e-3; `thw_s` / `ttc_s` / `lead_a_slope_mps2` carry only the definitional
  NaN of the lab rules.
* End-to-end check against a raw file (AstaZero platoon 4, Tesla Model 3 at position 2): `v_mps` equals `Speed2`,
  `lead_d_m` equals `IVS1`, `lead_speed_mps` equals `Speed1`, `lead_v_rel` equals `Speed1 - Speed2`, `das_enabled` equals
  `Driver2 == ACC` sample by sample; the derived acceleration correlates 0.90 with the central difference of the window
  speed, the lead slope 0.90 with the derivative of the lead speed.
* Physical ranges (finite samples; min / p0.1 / p99.9 / max): `openacc` v -0.07 / 0.00 / 38.9 / 46.3 m/s (111 samples
  of tiny negative GNSS speed at rest, within the training clip of -1), a -8.6 / -2.9 / 2.0 / 10.4 m/s^2 (2 samples
  above 10, see section 4), lead_d 0.5 / 0.8 / 120 / 150 m, v_rel -15.8 / -6.7 / 6.3 / 15.0 m/s, lead_speed 0 / 0 /
  38 / 46 m/s, slope +-4 (clipped), thw 0.02 / 0.30 / 11.8 / 118 s, ttc 0.07 / 1.8 / 515 / 1321 s (the scorer clips thw
  to 30 s and ttc to 60 s as in training); `openacc_500x` v 0 / 0 / 42.1 / 48.9 m/s, a -9.3 / -3.0 / 2.2 / 6.5 m/s^2.
* Window statistics: `openacc` moving share 0.97 (min per window 0.205), standstill 0.026, das 0.78, lead 0.75;
  `openacc_500x` moving 0.915 (min 0.202), standstill 0.067.
* Speed steps > 1.5 m/s per sample inside windows: 0 in both sets; frozen plateaus >= 3 s inside windows: 0.
* Identifier scan (`vtt.private` HEX12 / HEX16) of `vehicle_id`, `drive_id`, `note` and this file: no hits.

## 9. Caveats for the scorer and the write-up

1. Longitudinal-only input: nine channels (steering, yaw, curvature, lateral acceleration, pedals) are missing and lie
   outside the training augmentation -> score with `--mask-channels V2` (`openacc`) / `V3` (`openacc_500x`) or the
   reduced-input variants; the full model's output on these sets is a "reduced input" number at best.
2. `a_mps2` is a derivative of a GNSS (platoon) or interpolated OBD (500X) speed, not the vehicle's `aEgo`: noisier,
   with spline overshoot on hard manoeuvres (13 samples above 8 m/s^2); `standstill` is a 0.2 m/s threshold.
3. The platoon speeds of Cherasco / Vicolungo / Casale / JRC low speed are 10 Hz splines of 3-5 Hz GNSS, ZalaZone a
   B-spline of fused sensors; only AstaZero (RT-Range) is a true 10 Hz downsample. `openacc_500x` is interpolated OBD
   with 1/16 km/h quantisation and occasional holds (section 4).
4. ACC-driven: 78 % of the platoon samples are under ACC (AstaZero / ZalaZone / Casale by protocol); the longitudinal
   profile is the controller's, so compare with the lab's automation-engaged windows; `openacc_500x` is 100 % manual.
5. One physical car per model (two for the Tesla Model 3); vehicle and campaign are confounded (ZalaZone: 30-60 km/h
   proving ground, BEV-heavy; Cherasco / Vicolungo: motorway; AstaZero: rural track at 60-120 km/h) -> report per model
   and prefer within-campaign comparisons (AstaZero: Model 3 vs A6 / X5 / A-Class on identical laps; ZalaZone: five BEV
   vs Mazda 3 / GLE 450 / A4 on identical runs). A single vehicle gives no bootstrap interval.
6. Borderline footprints: Audi A8 (4.91), Audi e-tron (4.84), Jaguar I-Pace (4.92), Audi A4 Avant (4.41) -- 1,087 of
   the 4,540 windows; the scorer's "without borderline" size metrics are the robust ones here.
7. The two mild hybrids are labelled HEV (JRC spec file) to stay out of the BEV / ICE score as CANDIDATES.md asks;
   `--mhev-as-ice` relabels them ICE if the lab's mild-hybrid convention is to be applied, which adds 585 ICE windows
   (a LARGE and a SMALL car).
8. `das_enabled` for 94 windows rests on protocol assumptions (75 leader-human, 19 follower-ACC); the Casale leader and
   two JRC low speed S-Max runs may in truth have been on cruise control.
9. Lead availability depends on GNSS, not radar: no cut-ins, no far-range dropouts, and a lead beyond 150 m is declared
   absent by construction; the lead share (0.75) is above the training mean (0.54).
10. The JRC fleet merge (S-Max, 500X) is inferred from identical spec-file rows across campaigns, not documented by JRC;
    `--no-merge-fleet` reverts to per-campaign vehicle ids (22 vehicles instead of 21 in `openacc`).
11. Licence: CC BY 4.0 -- credit the European Commission JRC OpenACC database and state that the data were re-sampled,
    cleaned and re-labelled as described here.

## 10. Files

* Code: `experiments/transfer/ds_openacc.py` (adapter, QC, validation; `--help` lists the options; `--dry-run`
  validates without writing, `--limit-files N` for a quick pass).
* Data (never the raw folder): `external/openacc/work/{windows.npz, meta.parquet, windows_summary.json,
  qc_vehicle_files.csv, qc_summary.json, qc_tables.md, build.log}` and the same set under `external/openacc_500x/work/`.
* Labels: `experiments/transfer/specs_external.csv` (rows whose `datasets` column names OpenACC) and the campaign
  `*veh_specifications.csv` files under the raw campaign folders.
