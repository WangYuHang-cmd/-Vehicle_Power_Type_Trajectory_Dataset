# commaSteeringControl adapter (`ds_csc.py`): 36 platform archives -> transfer windows (`csc`)

Written 9 Nov 2026 by the csc adapter step of the zero-shot transfer campaign. Code `experiments/transfer/ds_csc.py`
(labels + build + validate), contract `common.py` / `CONTRACT.md`, labels through `ccs_platforms.Labeller`, raw data
`<data-root>/external/csc/data/<ARCHIVE>.zip` (36 archives, 20.9 GB zipped, downloaded by the owner from the Hugging Face
dataset `commaai/commaSteeringControl`, MIT), outputs `<data-root>/external/csc/work/`: `windows.npz`, `meta.parquet`,
`windows_summary.json` (the scorer's inputs), `csc_platforms.csv` (archive -> platform -> label table), `csc_inventory.csv`
(per archive: members, row-count histogram, eligible, read, kept, discards), `csc_segments_qc.csv` (one row per segment
read: pad / trim, status, speed and steering statistics, sign checks, EPS-firmware hash), `csc_validation.json`,
`csc_tables.md` (the tables of sections 3-4), `csc_build.log`, `parts/` (per-archive meta / qc / inventory).

```
source env.sh && nice -n 10 $CPY -I experiments/transfer/ds_csc.py build --workers 6 --cap 1500 --seed 0   # 28 min, disk-bound (20.9 GB of zips read twice: inventory + sample)
$CPY -I experiments/transfer/ds_csc.py validate                           # contract + physical-range checks of the saved set
$CPY -I experiments/transfer/ds_csc.py labels                             # the mapping table only
$CPY experiments/transfer/score.py --dataset csc --variant V1              # matching reduced-input models (see section 6)
$CPY experiments/transfer/score.py --dataset csc --mask-channels V1        # full models with the same mask, for comparison
```

## 1. Source

* commaSteeringControl (comma.ai, 2023): per-platform zip archives of 60-s segments selected by comma from openpilot
  routes in which the lateral controller was active. Inside each archive `<ARCHIVE>/NNNNN.csv`, one CSV per segment on an
  exact 10 Hz grid (`t` = 0.0, 0.1, ..., checked to 1 us on every segment read), 13 columns: `vEgo`, `aEgo`,
  `steeringAngleDeg`, `steeringPressed`, `steer`, `steerFiltered` (the EPS torque command and its filtered version),
  `latActive`, `roll`, `t`, `latAccelSteeringAngle` (comma's vehicle model from the steering angle, roll-compensated),
  `latAccelDesired`, `latAccelLocalizer` (localizer yaw rate x speed), `epsFwVersion`. No device, route, time or driver
  identifiers exist; the member index `NNNNN` is the only key.
* Segments are 598-601 rows long (599 rows is the most frequent length, section 4), a few are shorter (553-568) or longer
  (602-603). The data are highway-biased and automation-engaged (`latActive` true on about 92 % of the samples).
* The 36 archives are the 9 BEV archives named in `CANDIDATES.md` 4.2 plus 27 gasoline archives chosen by footprint class
  (SMALL / MID / LARGE) from the lab table. The archive names are the 2023 openpilot `CAR` strings with spaces, hyphens and
  parentheses replaced by underscores; in that naming every hybrid was a separate platform (`* HYBRID *`), so none of the
  36 archives contains hybrid cars (section 3).
* Untrusted input: the CSVs are read from the zips with `zipfile` + `pandas` (nothing is extracted or executed; members
  above 50 MB would be skipped as non-segments, none is). Reading happens with `python -I` so nothing next to the data
  can be imported.

## 2. Channel mapping

Every window has `channels_present = v_mps,a_mps2,steer_angle_deg,steer_rate_degps,yaw_rate_radps,curv_1pm,lat_acc_mps2,steer_pressed,standstill,das_enabled,lead_valid`.

| training channel | source column | status | how / caveat |
|---|---|---|---|
| `v_mps` | `vEgo` | measured | openpilot's filtered CAN speed, as in training (small negative values at rest, min -0.1 m/s, are openpilot's Kalman output and occur in the lab tables as well) |
| `a_mps2` | `aEgo` | measured | openpilot's speed-Kalman acceleration, as in training |
| `steer_angle_deg` | `steeringAngleDeg` | measured | steering wheel angle, left-positive (openpilot convention) |
| `steer_rate_degps` | d`steeringAngleDeg`/dt | **derived** | central difference of the 10 Hz angle (`np.gradient x 10`, one-sided at the window ends); the lab channel is the CAN steering-rate sample. Lab reference (one long drive per brand, 35 brands): corr(CAN rate, 10-Hz difference) median 0.97 (IQR 0.74-0.99), std ratio median 1.00; the difference is therefore a faithful substitute except on brands whose CAN rate is itself odd (Hyundai Elantra, Kia Stinger, Genesis G80, Volvo, Opel, Luxgen, Land Rover: corr < 0.1 in the lab, i.e. the lab channel, not the derivative, is the irregular one there) |
| `steer_torque_norm` | - | missing (NaN) | `steer` / `steerFiltered` are the EPS torque *command* of the controller, not the driver's torque; the torque group is absent (training augmentation state) |
| `yaw_rate_radps` | `-latAccelLocalizer / vEgo` | **derived** (sign flipped) | the data set's lateral acceleration is right-positive (device frame: corr(steeringAngleDeg, latAccelLocalizer) = -0.92 to -0.97 on every platform), the lab's yaw is left-positive (`POSE_YAW_SIGN = -1`, corr(steer, yaw) = +0.82 to +0.97 on the lab tables), hence the sign flip. Defined for v >= 1 m/s; **NaN below 1 m/s** (choice stated here: `latAccelLocalizer / v` is ill-conditioned at crawl speed, and the only steering-derived alternative, `latAccelSteeringAngle`, carries comma's roll-compensation term g sin(roll) that does not vanish with speed, so `latAccelSteeringAngle / v` diverges at v -> 0; the training yaw is NaN wherever the pose was stale, so NaN is inside the training vocabulary, and after standardisation NaN equals the training mean ~ 0, which is also the physical value at crawl speed) |
| `curv_1pm` | yaw / v | derived | `common.curvature_and_lat_acc`: yaw / v for v >= 1 m/s, NaN otherwise (lab rule) |
| `lat_acc_mps2` | `-latAccelLocalizer` | measured (sign flipped) | = v x yaw of the localizer, the same quantity as the lab's `v x yaw_rate`; defined at every speed |
| `gas_pressed`, `brake_pressed` | - | missing (NaN) | no pedal signals in the data set (pedal group absent) |
| `steer_pressed` | `steeringPressed` | measured | driver torque above the vehicle's threshold, as in training |
| `standstill` | `vEgo < 0.3 m/s` | **derived** | the lab flag is 0 at v >= 0.3 m/s on every brand checked and 1 on a median 97 % of the samples below 0.3 m/s (one long drive per brand, 35 brands) |
| `das_enabled` | `latActive` | **proxy** | lateral control active. The lab flag is controller enabled OR stock cruise enabled, which is also 1 while the controller is engaged but steering is inactive (below the car's minimum steering speed, during steering faults); `latActive` therefore under-counts the lab flag at low speed. Longitudinal state is unknown |
| `lead_valid` | constant 0 | absent group | the lead group is absent; 0 is the training augmentation's state for a dropped lead group (`CONTRACT.md`: `lead_valid` NaN would mean "unknown", a value no training window had) |
| `lead_d_m`, `lead_v_rel_mps`, `lead_speed_mps`, `lead_a_slope_mps2`, `thw_s`, `ttc_s` | - | missing (NaN) | no perception data |

The delivered state of the three absent groups equals `common.apply_channel_mask(X, "V1")` exactly. The task text asked
for "the whole lead group = NaN"; `lead_valid` is 0 instead for the reason in the table (the contract rule). The V1
checkpoints re-apply their own channel list at load time (`cfg["channels"]`, NaN -> 0), so for `--variant V1` the value of
`lead_valid` is irrelevant; for the full model with `--mask-channels V1` the scorer sets it to 0 as well.

Window assembly: one window per segment; 600 rows kept as they are; 598-599 rows -> the last row is repeated 1-2 times
(`note` carries `padded <k>`); 601 rows -> the last row is dropped; the rare 602-603-row files (whose `t` is still on the
grid) are trimmed to 600 the same way (`trimmed <k>`); shorter segments are discarded. Every window note also carries
`stratum=<seen|held-out>`, `label=lab`, `pt_lab=<lab label of the folded platform>` and the derivation rules. `vehicle_id` =
archive name (the data have no device ids, so one archive = one pseudo-vehicle for the cluster bootstrap), `drive_id` =
archive name + sha1 prefix of the member name, `w_idx` = 0, `t0_s` = 0.

## 3. Labels

### 3.1 Archive name -> lab platform

The archive names are the 2023 openpilot `CAR` strings (e.g. `HONDA CR-V 2017`, `HONDA CIVIC (BOSCH) 2019`,
`NISSAN LEAF 2018 Instrument Cluster`) with every non-alphanumeric character turned into an underscore. The current
opendbc keeps those strings as the keys of its `MIGRATION` table, so the fold is: strip non-alphanumerics on both sides,
look the archive up in `MIGRATION` (then among the current platform names). `ds_csc.py labels` does it with
`ccs_platforms.load_opendbc_names` (AST parse of the PyPI opendbc 0.3.1 tree of the decode venv, usable under `$CPY`); the
result was checked against `opendbc.car.fingerprints.MIGRATION` imported in the openpilot venv (`$PY`): identical for all 36
archives, no normalised-key collisions among the 279 MIGRATION entries. All 36 archives fold through `MIGRATION` (none needed
a manual rename); the folded platform is then looked up in the lab table `vtt/specs_data/platform_specs.csv`
(`ccs_platforms.Labeller`: footprint = wheelbase x mean track, model, spec confidence) and in `build/index/drives.parquet`
(powertrain label of the lab drives of that platform, split membership -> stratum). The full table (36 rows) is in
section 3.4; `specs_external.csv` holds 17 of the 36 keys (rows tagged `commaSteeringControl`), all with the lab table's
dimensions, and is used only for its `borderline` flag and as a cross-check (no disagreement on footprint or powertrain).

### 3.2 Footprint, size class, borderline

`footprint_m2` and `size_class` come from the lab table through the folded platform (so the same numbers the training
labels used; SMALL < 4.43 <= MID < 4.88 <= LARGE). Borderline = the strict rule |footprint - cut| < 0.05 m^2 OR the
`specs_external.csv` flag:

| archive(s) | footprint | class | why borderline |
|---|---|---|---|
| TOYOTA_CAMRY_2018, TOYOTA_CAMRY_2021 | 4.468 | MID | 0.038 above the 4.43 cut (strict rule) |
| HONDA_PILOT_2017 | 4.915 | LARGE | 0.035 above the 4.88 cut (strict rule; not in `specs_external.csv`) |
| HYUNDAI_IONIQ_5_2022 | 4.826 | MID | 0.054 below the cut, but Hyundai's 3,000 mm wheelbase gives 4.91 LARGE (`SPECS_EXTERNAL.md` conflict; the lab value is kept, as ruled in `CANDIDATES.md`) |

The scorer repeats the size metrics without the borderline models.

### 3.3 Powertrain

Rule: BEV when the lab table / drives index labels the folded platform BEV (9 archives; each archive name also carries a
BEV token: ELECTRIC, EV6, NIRO_EV, BOLT, LEAF, IONIQ_5 - the code asserts both agree), ICE otherwise. The ICE call rests on
the 2023 naming: hybrids were separate `CAR` values and separate archives (`MIGRATION` still lists the twins
`TOYOTA CAMRY HYBRID 2018 / 2021`, `TOYOTA COROLLA HYBRID TSS2 2019`, `TOYOTA HIGHLANDER HYBRID 2018 / 2020`,
`TOYOTA RAV4 HYBRID 2017 / 2019 / 2022`, `HONDA ACCORD HYBRID 2018`, `HYUNDAI TUCSON HYBRID 4TH GEN`), none of which was
downloaded, and the code refuses any archive whose name carries HYBRID / PHEV / PRIME / MIRAI / SYNTHETIC. The current
opendbc merged seven of those twins into one platform, and the lab labelled the merged platform by the ECU rule per route
(`vtt/labels.py ecu_resolve`), so the lab's platform-level label (mode over its drives) disagrees with the archive for
seven archives; the archive identity wins and the lab label is kept in `note` (`pt_lab=...`):

| archive | folded platform | lab drives (powertrain_raw counts) | label here |
|---|---|---|---|
| HONDA_ACCORD_2018 | HONDA_ACCORD | HEV 27 / ICE 16 / MIXED 6 | ICE |
| HYUNDAI_TUCSON_4TH_GEN | HYUNDAI_TUCSON_4TH_GEN | MIXED 175 / HEV 7 | ICE |
| TOYOTA_CAMRY_2018 | TOYOTA_CAMRY | MIXED 1 / ICE 1 / HEV 1 | ICE |
| TOYOTA_CAMRY_2021 | TOYOTA_CAMRY_TSS2 | HEV 29 / ICE 26 / MIXED 7 | ICE |
| TOYOTA_COROLLA_TSS2_2019 | TOYOTA_COROLLA_TSS2 | HEV 76 / MIXED 56 / ICE 30 | ICE |
| TOYOTA_RAV4_2022 | TOYOTA_RAV4_TSS2_2022 | HEV 11 / ICE 8 / MIXED 2 | ICE |
| (TOYOTA_HIGHLANDER_2020, TOYOTA_RAV4_2019) | TOYOTA_HIGHLANDER_TSS2, TOYOTA_RAV4_TSS2 | ICE-majority mixes (ICE 8 / HEV 7 / MIXED 5; ICE 34 / HEV 28 / MIXED 15) | ICE (agrees) |

Consequence for the BEV-vs-ICE score: the training data of those merged platforms contain hybrids labelled HEV (outside
the task) and gasoline cars labelled ICE; the CSC windows of the same platforms are gasoline only. For the Toyota TSS2
Corolla the lab model is held-out anyway.

### 3.4 Stratum

By car model against the drives index (`ccs_platforms.Labeller.stratum`): `seen` = the model has training or validation
drives, `held-out` = the model is one of the 41 `split_platform` test models, `absent` = not in the lab inventory. 26
archives are `seen` and 10 `held-out` (Genesis GV60, Hyundai Ioniq Electric, Hyundai Kona Electric, Hyundai Santa Fe
2019 and 2022, Nissan Leaf and Leaf IC, Toyota Corolla 2017 and TSS2 2019, Toyota Sienna); no archive is `absent`.
The scorer has no stratum column: the stratum is in every window's `note` (`stratum=...`) and in `csc_platforms.csv`, so a
stratified read-out filters `meta.note`.

### 3.5 The full mapping table (36 archives; `csc_platforms.csv`, rendered in `csc_tables.md`)

| archive (2023 name) | folded platform | via | model | footprint m^2 | class | borderline | powertrain | lab label of folded platform (drive mix) | stratum |
|---|---|---|---|---|---|---|---|---|---|
| CHEVROLET_BOLT_EUV_2022 | CHEVROLET_BOLT_EUV | MIGRATION | Chevrolet Bolt EUV | 3.986 | SMALL | no | BEV | BEV (BEV 73) | seen |
| CHEVROLET_SILVERADO_1500_2020 | CHEVROLET_SILVERADO | MIGRATION | Chevrolet Silverado 1500 | 6.649 | LARGE | no | ICE | ICE (ICE 196) | seen |
| GENESIS_GV60_ELECTRIC_1ST_GEN | GENESIS_GV60_EV_1ST_GEN | MIGRATION | Genesis GV60 | 4.734 | MID | no | BEV | BEV (BEV 75) | held-out |
| HONDA_ACCORD_2018 | HONDA_ACCORD | MIGRATION | Honda Accord | 4.542 | MID | no | ICE | HEV (HEV 27 / ICE 16 / MIXED 6) | seen |
| HONDA_CIVIC_2016 | HONDA_CIVIC | MIGRATION | Honda Civic | 4.197 | SMALL | no | ICE | ICE (ICE 19) | seen |
| HONDA_CIVIC_2022 | HONDA_CIVIC_2022 | MIGRATION | Honda Civic | 4.215 | SMALL | no | ICE | ICE (ICE 187) | seen |
| HONDA_CIVIC_BOSCH_2019 | HONDA_CIVIC_BOSCH | MIGRATION | Honda Civic | 4.197 | SMALL | no | ICE | ICE (ICE 692) | seen |
| HONDA_CR_V_2016 | HONDA_CRV | MIGRATION | Honda CR-V | 4.227 | SMALL | no | ICE | ICE (ICE 8) | seen |
| HONDA_CR_V_2017 | HONDA_CRV_5G | MIGRATION | Honda CR-V | 4.277 | SMALL | no | ICE | ICE (ICE 18) | seen |
| HONDA_ODYSSEY_2018 | HONDA_ODYSSEY | MIGRATION | Honda Odyssey | 5.145 | LARGE | no | ICE | ICE (ICE 19) | seen |
| HONDA_PILOT_2017 | HONDA_PILOT | MIGRATION | Honda Pilot | 4.915 | LARGE | yes | ICE | ICE (ICE 17) | seen |
| HYUNDAI_ELANTRA_2021 | HYUNDAI_ELANTRA_2021 | MIGRATION | Hyundai Elantra | 4.311 | SMALL | no | ICE | ICE (ICE 73) | seen |
| HYUNDAI_IONIQ_5_2022 | HYUNDAI_IONIQ_5 | MIGRATION | Hyundai Ioniq 5 | 4.826 | MID | yes | BEV | BEV (BEV 186) | seen |
| HYUNDAI_IONIQ_ELECTRIC_LIMITED_2019 | HYUNDAI_IONIQ_EV_LTD | MIGRATION | Hyundai Ioniq | 4.217 | SMALL | no | BEV | BEV (BEV 3) | held-out |
| HYUNDAI_KONA_ELECTRIC_2019 | HYUNDAI_KONA_EV | MIGRATION | Hyundai Kona | 4.025 | SMALL | no | BEV | BEV (BEV 27) | held-out |
| HYUNDAI_PALISADE_2020 | HYUNDAI_PALISADE | MIGRATION | Hyundai Palisade | 4.965 | LARGE | no | ICE | ICE (ICE 45) | seen |
| HYUNDAI_SANTA_FE_2019 | HYUNDAI_SANTA_FE | MIGRATION | Hyundai Santa Fe | 4.531 | MID | no | ICE | ICE (ICE 8) | held-out |
| HYUNDAI_SANTA_FE_2022 | HYUNDAI_SANTA_FE_2022 | MIGRATION | Hyundai Santa Fe | 4.567 | MID | no | ICE | ICE (ICE 13) | held-out |
| HYUNDAI_SONATA_2019 | HYUNDAI_SONATA_LF | MIGRATION | Hyundai Sonata | 4.565 | MID | no | ICE | ICE (ICE 1) | seen |
| HYUNDAI_SONATA_2020 | HYUNDAI_SONATA | MIGRATION | Hyundai Sonata | 4.624 | MID | no | ICE | ICE (ICE 24) | seen |
| HYUNDAI_TUCSON_4TH_GEN | HYUNDAI_TUCSON_4TH_GEN | MIGRATION | Hyundai Tucson | 4.527 | MID | no | ICE | MIXED (MIXED 175 / HEV 7) | seen |
| KIA_EV6_2022 | KIA_EV6 | MIGRATION | Kia EV6 | 4.744 | MID | no | BEV | BEV (BEV 150) | seen |
| KIA_NIRO_EV_2020 | KIA_NIRO_EV | MIGRATION | Kia Niro | 4.267 | SMALL | no | BEV | BEV (BEV 154) | seen |
| NISSAN_LEAF_2018 | NISSAN_LEAF | MIGRATION | Nissan Leaf | 4.160 | SMALL | no | BEV | BEV (BEV 49) | held-out |
| NISSAN_LEAF_2018_Instrument_Cluster | NISSAN_LEAF_IC | MIGRATION | Nissan Leaf | 4.160 | SMALL | no | BEV | BEV (BEV 13) | held-out |
| RAM_1500_5TH_GEN | RAM_1500_5TH_GEN | MIGRATION | RAM 1500 | 6.732 | LARGE | no | ICE | ICE (ICE 46) | seen |
| TOYOTA_CAMRY_2018 | TOYOTA_CAMRY | MIGRATION | Toyota Camry | 4.468 | MID | yes | ICE | HEV (MIXED 1 / ICE 1 / HEV 1) | seen |
| TOYOTA_CAMRY_2021 | TOYOTA_CAMRY_TSS2 | MIGRATION | Toyota Camry | 4.468 | MID | yes | ICE | HEV (HEV 29 / ICE 26 / MIXED 7) | seen |
| TOYOTA_COROLLA_2017 | TOYOTA_COROLLA | MIGRATION | Toyota Corolla | 4.139 | SMALL | no | ICE | ICE (ICE 5) | held-out |
| TOYOTA_COROLLA_TSS2_2019 | TOYOTA_COROLLA_TSS2 | MIGRATION | Toyota Corolla | 4.093 | SMALL | no | ICE | HEV (HEV 76 / MIXED 56 / ICE 30) | held-out |
| TOYOTA_HIGHLANDER_2017 | TOYOTA_HIGHLANDER | MIGRATION | Toyota Highlander | 4.603 | MID | no | ICE | ICE (ICE 5 / MIXED 3) | seen |
| TOYOTA_HIGHLANDER_2020 | TOYOTA_HIGHLANDER_TSS2 | MIGRATION | Toyota Highlander | 4.680 | MID | no | ICE | ICE (ICE 8 / HEV 7 / MIXED 5) | seen |
| TOYOTA_RAV4_2017 | TOYOTA_RAV4 | MIGRATION | Toyota RAV4 | 4.134 | SMALL | no | ICE | ICE (ICE 16) | seen |
| TOYOTA_RAV4_2019 | TOYOTA_RAV4_TSS2 | MIGRATION | Toyota RAV4 | 4.344 | SMALL | no | ICE | ICE (ICE 34 / HEV 28 / MIXED 15) | seen |
| TOYOTA_RAV4_2022 | TOYOTA_RAV4_TSS2_2022 | MIGRATION | Toyota RAV4 | 4.344 | SMALL | no | ICE | HEV (HEV 11 / ICE 8 / MIXED 2) | seen |
| TOYOTA_SIENNA_2018 | TOYOTA_SIENNA | MIGRATION | Toyota Sienna | 5.212 | LARGE | no | ICE | ICE (ICE 9) | held-out |

## 4. Counts

* **Inventory.** 390,427 segments in the 36 archives (500-25,000 per archive). Row counts: 598 rows 3,283 (0.8 %), 599 rows 196,747 (50.4 %), 600 rows 186,992 (47.9 %), 601 rows 2,326 (0.6 %), shorter than 598 rows 980 (0.3 %, down to 204 rows; discarded), longer than 601 rows 99 (up to 720 rows; trimmed). Eligible (>= 598 rows): 389,447.
* **Sampling.** Cap 1,500 windows per archive in a seeded random order (`numpy.random.default_rng(0)` over the eligible members of each archive): 31 archives hit the cap, 5 have fewer eligible segments than the cap and contribute every eligible one (GENESIS_GV60_ELECTRIC_1ST_GEN 999 of 1,000; HONDA_CR_V_2016 1,000 of 1,000; HYUNDAI_IONIQ_ELECTRIC_LIMITED_2019 497 of 500; HYUNDAI_TUCSON_4TH_GEN 999 of 1,000; NISSAN_LEAF_2018_Instrument_Cluster 1,467 of 1,500). 51,463 segments read, 51,462 windows kept, 1 discarded (stationary 1: fewer than 20 % moving samples, the lab window rule). No segment failed the grid, column or NaN checks (`t` on the 10 Hz grid to 1 us everywhere; no NaN in any numeric column of the 51,463 segments read).
* **Pad / trim.** 26,202 windows padded (50.9 %: 25,770 by one row, 432 by two), 320 trimmed (0.6 %: 304 by one row, 16 by 2-11 rows).
* **Result.** 51,462 windows = 857.7 h, 36 pseudo-vehicles (archives), 36 lab platforms, 25 car models. EPS-firmware versions per archive (a lower bound on distinct cars, from every member of the archive): 1-17, median 4.

| group | value | archives | models | windows | hours |
|---|---|---|---|---|---|
| size_class | LARGE | 6 | 6 | 9,000 | 150.0 |
| size_class | MID | 13 | 9 | 18,498 | 308.3 |
| size_class | SMALL | 17 | 10 | 23,964 | 399.4 |
| powertrain | BEV | 9 | 8 | 11,963 | 199.4 |
| powertrain | ICE | 27 | 17 | 39,499 | 658.3 |
| stratum | held-out | 10 | 7 | 13,463 | 224.4 |
| stratum | seen | 26 | 18 | 37,999 | 633.3 |
| class x powertrain | LARGE ICE | 6 | 6 | 9,000 | 150.0 |
| class x powertrain | MID BEV | 3 | 3 | 3,999 | 66.7 |
| class x powertrain | MID ICE | 10 | 6 | 14,499 | 241.7 |
| class x powertrain | SMALL BEV | 6 | 5 | 7,964 | 132.7 |
| class x powertrain | SMALL ICE | 11 | 5 | 16,000 | 266.7 |

Per archive (`csc_inventory.csv` + `csc_validation.json`; `read` = segments parsed, `kept` = windows; `das %` = share of samples with `latActive`; `corr(steer, yaw)` = correlation of the steering angle with the derived yaw rate over samples above 5 m/s, the sign check of section 2):

| archive | class | powertrain | stratum | segments | rows 598 / 599 / 600 / 601 / other | eligible (>= 598) | read | kept | padded | trimmed | discards | EPS fw (all) | hours | v mean m/s | moving % | v<0.3 % | das % | steer pressed % | corr(steer, yaw) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CHEVROLET_BOLT_EUV_2022 | SMALL | BEV | seen | 25,000 | 256 / 12570 / 11924 / 140 / 110 | 24,891 | 1,500 | 1,500 | 787 | 12 | - | 1 | 25.0 | 24.0 | 98.7 | 1.1 | 90.7 | 7.2 | 0.93 |
| CHEVROLET_SILVERADO_1500_2020 | LARGE | ICE | seen | 7,000 | 50 / 3587 / 3281 / 38 / 44 | 6,959 | 1,500 | 1,500 | 749 | 6 | - | 1 | 25.0 | 27.1 | 98.9 | 0.9 | 92.4 | 5.9 | 0.91 |
| GENESIS_GV60_ELECTRIC_1ST_GEN | MID | BEV | held-out | 1,000 | 4 / 511 / 477 / 7 / 1 | 999 | 999 | 999 | 515 | 7 | - | 1 | 16.6 | 27.8 | 99.0 | 0.6 | 94.9 | 2.9 | 0.85 |
| HONDA_ACCORD_2018 | MID | ICE | seen | 24,999 | 298 / 12656 / 11863 / 116 / 66 | 24,944 | 1,500 | 1,500 | 784 | 4 | - | 7 | 25.0 | 26.1 | 98.1 | 1.4 | 91.9 | 8.9 | 0.92 |
| HONDA_CIVIC_2016 | SMALL | ICE | seen | 16,998 | 109 / 8575 / 8118 / 118 / 78 | 16,926 | 1,500 | 1,500 | 833 | 8 | - | 5 | 25.0 | 23.1 | 97.6 | 1.9 | 91.9 | 10.2 | 0.94 |
| HONDA_CIVIC_2022 | SMALL | ICE | seen | 4,500 | 47 / 2328 / 2085 / 27 / 13 | 4,487 | 1,500 | 1,500 | 802 | 11 | - | 4 | 25.0 | 24.1 | 97.9 | 1.6 | 91.6 | 5.7 | 0.91 |
| HONDA_CIVIC_BOSCH_2019 | SMALL | ICE | seen | 14,000 | 131 / 7009 / 6722 / 83 / 55 | 13,952 | 1,501 | 1,500 | 766 | 9 | stationary 1 | 10 | 25.0 | 23.8 | 97.8 | 1.8 | 90.4 | 12.9 | 0.94 |
| HONDA_CR_V_2016 | SMALL | ICE | seen | 1,000 | 7 / 520 / 470 / 3 / 0 | 1,000 | 1,000 | 1,000 | 527 | 3 | - | 1 | 16.7 | 24.9 | 99.2 | 0.6 | 88.7 | 0.0 | 0.94 |
| HONDA_CR_V_2017 | SMALL | ICE | seen | 17,000 | 199 / 8562 / 8133 / 75 / 31 | 16,975 | 1,500 | 1,500 | 751 | 7 | - | 5 | 25.0 | 24.7 | 97.8 | 1.6 | 92.2 | 12.9 | 0.94 |
| HONDA_ODYSSEY_2018 | LARGE | ICE | seen | 14,987 | 122 / 7494 / 7254 / 100 / 17 | 14,971 | 1,500 | 1,500 | 749 | 10 | - | 3 | 25.0 | 27.5 | 99.1 | 0.7 | 91.6 | 8.1 | 0.91 |
| HONDA_PILOT_2017 | LARGE | ICE | seen | 12,500 | 94 / 6440 / 5880 / 73 / 13 | 12,488 | 1,500 | 1,500 | 773 | 9 | - | 6 | 25.0 | 25.8 | 99.3 | 0.6 | 92.9 | 13.3 | 0.91 |
| HYUNDAI_ELANTRA_2021 | SMALL | ICE | seen | 5,000 | 42 / 2383 / 2537 / 18 / 20 | 4,981 | 1,500 | 1,500 | 731 | 7 | - | 6 | 25.0 | 24.4 | 97.9 | 1.7 | 90.6 | 5.5 | 0.94 |
| HYUNDAI_IONIQ_5_2022 | MID | BEV | seen | 8,500 | 44 / 3444 / 4950 / 32 / 30 | 8,470 | 1,500 | 1,500 | 621 | 7 | - | 4 | 25.0 | 23.5 | 97.8 | 1.8 | 92.3 | 7.1 | 0.92 |
| HYUNDAI_IONIQ_ELECTRIC_LIMITED_2019 | SMALL | BEV | held-out | 500 | 1 / 258 / 238 / 0 / 3 | 497 | 497 | 497 | 259 | 0 | - | 1 | 8.3 | 21.1 | 99.9 | 0.0 | 92.8 | 4.1 | 0.91 |
| HYUNDAI_KONA_ELECTRIC_2019 | SMALL | BEV | held-out | 2,500 | 35 / 1256 / 1191 / 17 / 1 | 2,500 | 1,500 | 1,500 | 791 | 12 | - | 3 | 25.0 | 24.1 | 98.8 | 0.9 | 92.6 | 4.9 | 0.94 |
| HYUNDAI_PALISADE_2020 | LARGE | ICE | seen | 13,999 | 92 / 6914 / 6891 / 80 / 22 | 13,979 | 1,500 | 1,500 | 764 | 12 | - | 7 | 25.0 | 25.8 | 98.8 | 0.9 | 93.3 | 7.1 | 0.90 |
| HYUNDAI_SANTA_FE_2019 | MID | ICE | held-out | 8,499 | 91 / 4306 / 4026 / 61 / 15 | 8,489 | 1,500 | 1,500 | 812 | 11 | - | 4 | 25.0 | 25.4 | 98.8 | 0.9 | 92.5 | 5.4 | 0.92 |
| HYUNDAI_SANTA_FE_2022 | MID | ICE | held-out | 4,500 | 36 / 2098 / 2337 / 21 / 8 | 4,492 | 1,500 | 1,500 | 702 | 7 | - | 1 | 25.0 | 23.5 | 97.8 | 1.8 | 93.3 | 4.4 | 0.93 |
| HYUNDAI_SONATA_2019 | MID | ICE | seen | 2,000 | 12 / 1020 / 951 / 10 / 7 | 1,993 | 1,500 | 1,500 | 781 | 8 | - | 1 | 25.0 | 22.9 | 97.3 | 2.3 | 89.8 | 5.1 | 0.94 |
| HYUNDAI_SONATA_2020 | MID | ICE | seen | 21,000 | 161 / 10156 / 10511 / 131 / 41 | 20,965 | 1,500 | 1,500 | 735 | 5 | - | 14 | 25.0 | 23.9 | 98.0 | 1.6 | 91.6 | 6.0 | 0.92 |
| HYUNDAI_TUCSON_4TH_GEN | MID | ICE | seen | 1,000 | 6 / 463 / 522 / 8 / 1 | 999 | 999 | 999 | 469 | 8 | - | 3 | 16.6 | 23.8 | 99.2 | 0.7 | 93.2 | 7.4 | 0.91 |
| KIA_EV6_2022 | MID | BEV | seen | 4,500 | 20 / 1766 / 2684 / 17 / 13 | 4,487 | 1,500 | 1,500 | 563 | 9 | - | 6 | 25.0 | 21.7 | 97.4 | 2.0 | 93.1 | 8.1 | 0.91 |
| KIA_NIRO_EV_2020 | SMALL | BEV | seen | 8,000 | 82 / 3920 / 3902 / 56 / 40 | 7,966 | 1,500 | 1,500 | 752 | 11 | - | 4 | 25.0 | 22.8 | 98.3 | 1.3 | 90.4 | 7.3 | 0.94 |
| NISSAN_LEAF_2018 | SMALL | BEV | held-out | 12,000 | 70 / 6072 / 5716 / 65 / 77 | 11,926 | 1,500 | 1,500 | 771 | 9 | - | 5 | 25.0 | 20.9 | 97.6 | 2.0 | 90.1 | 11.6 | 0.93 |
| NISSAN_LEAF_2018_Instrument_Cluster | SMALL | BEV | held-out | 1,500 | 14 / 746 / 700 / 6 / 34 | 1,467 | 1,467 | 1,467 | 760 | 7 | - | 3 | 24.4 | 21.8 | 99.4 | 0.4 | 91.9 | 16.1 | 0.91 |
| RAM_1500_5TH_GEN | LARGE | ICE | seen | 13,991 | 129 / 7015 / 6752 / 75 / 20 | 13,972 | 1,500 | 1,500 | 786 | 8 | - | 17 | 25.0 | 28.3 | 99.7 | 0.2 | 94.1 | 10.2 | 0.84 |
| TOYOTA_CAMRY_2018 | MID | ICE | seen | 13,000 | 131 / 6605 / 6165 / 81 / 18 | 12,984 | 1,500 | 1,500 | 775 | 18 | - | 6 | 25.0 | 26.6 | 99.2 | 0.6 | 91.7 | 6.1 | 0.89 |
| TOYOTA_CAMRY_2021 | MID | ICE | seen | 6,000 | 44 / 3077 / 2834 / 36 / 9 | 5,991 | 1,500 | 1,500 | 781 | 8 | - | 2 | 25.0 | 25.7 | 98.7 | 1.0 | 92.7 | 6.2 | 0.91 |
| TOYOTA_COROLLA_2017 | SMALL | ICE | held-out | 11,498 | 99 / 6144 / 5145 / 80 / 30 | 11,470 | 1,500 | 1,500 | 839 | 11 | - | 2 | 25.0 | 25.1 | 99.0 | 0.9 | 90.6 | 6.7 | 0.91 |
| TOYOTA_COROLLA_TSS2_2019 | SMALL | ICE | held-out | 24,998 | 203 / 13245 / 11322 / 156 / 72 | 24,933 | 1,500 | 1,500 | 816 | 11 | - | 10 | 25.0 | 23.2 | 97.5 | 1.9 | 92.1 | 6.2 | 0.92 |
| TOYOTA_HIGHLANDER_2017 | MID | ICE | seen | 7,973 | 38 / 4200 / 3680 / 34 / 21 | 7,954 | 1,500 | 1,500 | 797 | 11 | - | 2 | 25.0 | 24.9 | 98.5 | 1.2 | 93.5 | 6.8 | 0.89 |
| TOYOTA_HIGHLANDER_2020 | MID | ICE | seen | 23,999 | 161 / 12468 / 11136 / 165 / 69 | 23,938 | 1,500 | 1,500 | 759 | 11 | - | 5 | 25.0 | 24.0 | 98.2 | 1.4 | 92.6 | 8.3 | 0.92 |
| TOYOTA_RAV4_2017 | SMALL | ICE | seen | 12,000 | 96 / 6121 / 5670 / 94 / 19 | 11,983 | 1,500 | 1,500 | 779 | 12 | - | 3 | 25.0 | 27.5 | 99.5 | 0.4 | 93.0 | 6.9 | 0.86 |
| TOYOTA_RAV4_2019 | SMALL | ICE | seen | 25,000 | 199 / 12889 / 11701 / 178 / 33 | 24,976 | 1,500 | 1,500 | 799 | 10 | - | 7 | 25.0 | 23.9 | 97.6 | 1.8 | 91.6 | 6.7 | 0.91 |
| TOYOTA_RAV4_2022 | SMALL | ICE | seen | 8,999 | 72 / 4496 / 4341 / 59 / 31 | 8,968 | 1,500 | 1,500 | 750 | 8 | - | 2 | 25.0 | 24.8 | 98.9 | 0.8 | 93.2 | 6.2 | 0.91 |
| TOYOTA_SIENNA_2018 | LARGE | ICE | held-out | 10,487 | 88 / 5433 / 4883 / 66 / 17 | 10,475 | 1,500 | 1,500 | 774 | 13 | - | 3 | 25.0 | 25.4 | 98.7 | 1.0 | 93.1 | 8.8 | 0.90 |
| total | | | | 390,427 | | 389,447 | 51,463 | 51,462 | 26202 | 320 | | | 857.7 | | | | | | |

## 5. Validation (`ds_csc.py validate`, 0 problems; `csc_validation.json`)

* Shape (51,462, 600, 20) float32; `channels_present` identical on every window; no NaN in the nine present channels (`v_mps, a_mps2, steer_angle_deg, steer_rate_degps, lat_acc_mps2, steer_pressed, standstill, das_enabled, lead_valid`); `yaw_rate_radps` and `curv_1pm` are NaN on exactly the samples with v < 1 m/s (1.6 % of the samples); the nine absent channels are NaN everywhere and `lead_valid` is 0 everywhere; flags are 0 / 1; `standstill == (v < 0.3 m/s)` on every sample; every window has `moving_frac` >= 0.2 and `meta.moving_frac` / `meta.das_frac` match X; `(drive_id, w_idx)` unique, one window per drive; one model / platform / class / powertrain / footprint per vehicle; labels of every window equal the label table; powertrain vocabulary {BEV, ICE}; no identifier-like token or path in any meta string.
* **Sign check.** Correlation of the steering angle with the derived (left-positive) yaw rate above 5 m/s: per archive min 0.84 (RAM 1500), median 0.91, no archive below 0.5; per segment (`csc_segments_qc.csv`, `corr_steer_lat`) median 0.95, 5 of 45,940 negative (low-speed manoeuvres). The two raw lateral-acceleration columns agree with each other (corr(latAccelSteeringAngle, latAccelLocalizer) median 0.98), so the sign flip applies to the localizer frame, not to a per-brand quirk.
* **Speed.** -0.1 to 51.6 m/s (max 185.6 km/h, two windows above 45 m/s, an Accord and an EV6), mean 24.6, median 27.0, 99 % 37.1; moving share 98.5 %, below 0.3 m/s 1.2 %, exactly 0 0.4 % (lab eligible windows: mean speed 17.6 m/s, moving 90.1 %, standstill 7.5 %). Small negative speeds (min -0.13 m/s) are openpilot's Kalman output at rest.
* **Acceleration.** std 0.41 m/s^2, 1 / 99 % quantiles -1.58 / 1.28; |a| > 5 m/s^2 on 0.002 % of the samples; |a| > 8 m/s^2 (the training clip bound) on 36 samples (1.2e-6 of the samples; 18 windows of 15 archives; min -16.6, max 10.4): isolated 0.1-s `aEgo` spikes with speed steps of up to 4.4 m/s between consecutive samples, i.e. CAN glitches that the scorer's clip removes; they are left in the windows (the contract forbids clipping in the adapter) and reported here (`validate` flags the tail only above 0.01 % of the samples).
* **Lateral channels.** Steering angle std 13.4 deg over all samples (per-archive std 13.1 deg on average; 99.9 % of |angle| below 199.5 deg; one sample above 540 deg), derived rate std 8.2 deg/s (99.9 % of |rate| below 120.4 deg/s; 88 samples above 500 deg/s, parking manoeuvres), lateral acceleration std 0.51 m/s^2 (max |lat| 10.8), yaw rate std 0.033 rad/s (9 samples above 1 rad/s), curvature std 0.0050 1/m (max |curv| 0.26).
* **Automation flag.** `das_frac` mean 92.1 %, median 100.0 %, 10 % quantile 66.3 %; 67.3 % of the windows fully engaged, none fully disengaged: every CSC window is an automation-engaged window.
* **Per-archive consistency.** Mean speed per archive 20.9-28.3 m/s, moving share 97.3-99.9 %, `das` share 88.7-94.9 %, steering-pressed share 0.0-16.1 % (HONDA_CR_V_2016 is the outlier: `steeringPressed` is set in only 3 of its 1,000 segments, at most 1 % of a segment, while the lab's eight HONDA_CRV drives show a median share of 1.0 % and a maximum of 15.6 %; the flag exists on this platform, overrides are just absent from the engaged segments comma selected).

## 6. How to score, and caveats

* **Matching reduced-input models: V1** (lateral + speed, 10 of 20 channels: `v_mps, a_mps2, standstill, das_enabled,
  steer_angle_deg, steer_rate_degps, yaw_rate_radps, curv_1pm, lat_acc_mps2, steer_pressed`; `VARIANTS.md`):
  `score.py --dataset csc --variant V1`. Internal references on the lab's held-out models (drive level, macro-F1): size
  V1 66.1 (full 67.1), BEV vs ICE V1 74.2 (full 93.0): the powertrain task loses most of its signal without the lead,
  pedal and torque channels, so a CSC BEV-vs-ICE number must be read against 74, not 93. The full models with
  `--mask-channels V1` are the secondary number (same mask applied to the 20-channel model).
* **Compare with the lab's automation-engaged windows.** `das_enabled` = `latActive` is 1 on about 92 % of the CSC
  samples (lab eligible windows: `das_frac` mean 0.55, 57 % `das` kind, 20 % human-only), and CSC is highway-biased
  (mean speed 24.6 m/s against 17.6 m/s for the lab windows, share of samples below 0.3 m/s 1.2 % against a standstill share of 7.5 %). The human-only
  protocol does not apply.
* **One pseudo-vehicle per archive.** Without device ids, every archive is one `vehicle_id`; the vehicle-cluster bootstrap
  therefore resamples archives, and the two Leaf archives (`NISSAN_LEAF_2018`, `_Instrument_Cluster`) are the same car
  model twice. The EPS-firmware strings give a lower bound on the number of distinct cars per archive (`csc_inventory.csv`,
  column `n_eps_fw_all`; 1-17 firmware versions per archive, median 4), not a vehicle id.
* **Overlap with training routes cannot be excluded**: comma drew the segments from the openpilot fleet, the lab's raw
  library comes from the same fleet, and no route key exists. The `seen` stratum (26 archives) may therefore contain
  training routes; the `held-out` stratum (10 archives, 7 models) is clean with respect to the lab's split by
  construction only if comma's and the lab's devices differ, which is unknown. Treat `seen` results as an upper bound.
* **Derived lateral channels**: steering rate from a 10-Hz difference, yaw from the localizer lateral acceleration (NaN
  below 1 m/s), `das_enabled` from the lateral-active flag; `steer_torque_norm` and both pedal flags are absent. Every
  deviation is stated in `meta.note`.
* **Padded windows**: about half of the segments are 599 rows long and get their last row repeated once (`padded 1`),
  a 0.1-s hold at the end of the window (50.9 % of the windows); `padded 2` is rare (598 rows, 0.8 %).

