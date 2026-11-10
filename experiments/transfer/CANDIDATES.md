# External datasets for the transfer test: candidates

Synthesis of the scout and verification passes, 9 Nov 2026. Purpose: public datasets on which the final size-class and
powertrain models can be tested zero-shot, outside the lab's own openpilot logs. Section 1 ranks the 14 verified sets,
section 2 gives adapter plans for the top 5, section 3 lists the unverified long tail, sections 4-5 say who downloads
what. All sizes, rates and durations below were measured or confirmed against primary sources during verification; the
few places where documentation and files disagree are flagged.

## 0. What the models need (reference)

* Window: 60 s at 10 Hz (600 samples) cut from a continuous recording; one prediction per window, aggregated per drive
  and per vehicle.
* Official inputs (20 channels): `v_mps, a_mps2, steer_angle_deg, steer_rate_degps, steer_torque_norm, yaw_rate_radps,
  curv_1pm, lat_acc_mps2, gas_pressed, brake_pressed, steer_pressed, standstill, das_enabled, lead_valid, lead_d_m,
  lead_v_rel_mps, lead_speed_mps, lead_a_slope_mps2, thw_s, ttc_s`.
  * Training definitions to reproduce: `curv_1pm = yaw / v` (v >= 1 m/s), `lat_acc_mps2 = v * yaw`, yaw from the device
    IMU/pose (not from steering); `steer_torque_norm` = driver torque divided by a per-drive quantile of |torque| over
    moving samples with the driver steering (no platform constant needed, so it works on unseen cars);
    `lead_speed_mps = v + v_rel`; `thw_s = d / v`; `ttc_s = -d / v_rel` while closing (clipped 0-60 s);
    `das_enabled` = automation enabled OR stock cruise enabled.
  * Missing groups: the lead group carries its own flag (`lead_valid`; vision-only cars already have no radar lead in
    training) and the training augmentation drops one group at a time among lead / torque / pedals. Missing lateral
    channels (steering, yaw, curvature, lateral acceleration) are covered by no augmentation, so datasets without them
    need a longitudinal-only model variant (see 2.3).
* Labels: size = footprint (wheelbase x mean track) from the lab platform spec table, thresholds 4.43 / 4.88 m^2
  (SMALL < 4.43 <= MID < 4.88 <= LARGE); powertrain task = BEV vs ICE only (HEV, PHEV, FCEV excluded).
  For named models absent from the lab table the footprint comes from public spec sheets (marked "public" below,
  about +-0.05 m^2).
* Headline split: 41 car models were held out of training, among them Tesla Model S and Model Y, Nissan Leaf, Genesis
  GV60 / GV70 / GV80, Ford F-150 Lightning, VW ID.3 / ID.4, Hyundai Kona and Ioniq, Lexus RX, Toyota Corolla, Kia EV9 /
  Sorento / Stinger. External windows from these models test unseen model AND unseen device at once.

Fit grades used below: **full** = most of the 20 channels at >= 10 Hz, several labelled models, both classes of the
task, so the metric is comparable with the internal test; **partial** = a channel subset (needs a masked or reduced
variant) or few models / one class thin, so only per-model or reduced-input accuracy; **sanity** = one vehicle or one
class, so only the expected-label hit rate.

Download column: **OWNER** = registration, agreement, or more than about 2 GB; **ASSISTANT** = open and small enough
to script in-session. None of the 14 verified sets needs registration: the OWNER list is driven by volume (and one
browser-only download button). Lab rule unchanged: the owner runs every download; the column only says which pulls are
small enough to be done in-session on request.

## 1. Ranked table (verified)

| # | dataset | link | vehicles and labels | signals and rate | continuity | access | size | size task | powertrain task | download method | who |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | commaCarSegments (comma.ai) | https://huggingface.co/datasets/commaai/commaCarSegments | ~188 opendbc platforms (230 keys incl. 42 legacy spellings), 188,883 x 60-s segments, 5,146 devices, 101,471 routes; 20 BEV platforms = 28,121 segments / 845 devices; label = platform name (carParams.carFingerprint, same naming as the lab table); VIN prefix on most brands; no driver metadata | raw CAN 100 Hz + carParams + pandaStates 10 Hz; opendbc decode gives vEgo, aEgo, steering angle / rate / torque, pedal and steering flags, standstill, cruise at 100 Hz; radar lead 20 Hz on radar cars (Toyota/Lexus, radar Honda, Hyundai-Kia-Genesis CAN SCC, GM radar, Chrysler/Ram/Jeep, Ford); no IMU, GPS, video | each segment one gap-free 60 s; only 24 % of segments sit in consecutive runs, so treat segments as independent windows | open, MIT, no login | 298.6 GB total (1.58 MB / segment); 50 platforms x 400 segments = ~32 GB | **full** (all three classes, 53 platforms with >= 1,000 segments); yaw / curvature / lateral accel must be steering-derived on most brands; same fleet: drop 464 devices shared with the lab | **full** (20 BEV vs ~150 ICE platforms, hybrids excluded; 845 BEV devices); most BEVs are vision-only, so the BEV decision runs without lead features | index JSON + per-segment HTTPS (curl / `hf download --include`) | OWNER |
| 2 | commaSteeringControl (comma.ai) | https://huggingface.co/datasets/commaai/commaSteeringControl | 119 real-car platform archives (118 platforms, 18 brands): 9 BEV archives / 8 models, 76 ICE, 33 HEV-PHEV, 1 FCEV, 1 synthetic (exclude); label = archive name in 2023 openpilot naming (~45 manual renames to the lab table); no device, route, time or driver ids | 13 columns on an exact 10 Hz grid: vEgo, aEgo, steeringAngleDeg, steeringPressed, latActive, roll, latAccelSteeringAngle / Desired / Localizer, steer command, EPS firmware; no pedals, no lead, no driver torque, no IMU | fixed ~60-s segments (598-601 rows), not stitchable, no route key | open, MIT | 45.5 GB zipped (26 MB-1.4 GB per platform; BEV set 3.3 GB) -> ~133 GB CSV, ~0.85 M segments | **partial** (speed + lateral channels only; lead, torque and pedal groups all masked; engaged-only highway-biased windows; overlap with training routes cannot be excluded) | **partial** (8 BEV vs 76 ICE models on speed / accel / lateral channels only) | `hf download --include data/<PLATFORM>.zip` | OWNER for the set; ASSISTANT may pull single platform zips for a pilot |
| 3 | ULTra-AV (CATS Lab, Sci. Data 2024) | https://doi.org/10.6084/m9.figshare.26339512 | harmonised car-following CSVs from 14 sources; ~20 named follower models, one physical car each: 6 BEV (Tesla Model 3 / S / X, Audi e-tron, Jaguar I-Pace, VW e-Golf), 7 ICE (Audi A6, BMW X5, Mercedes A-Class, Mazda 3, Ford S-Max, Peugeot 5008, Toyota Corolla), 9 hybrid / FCEV / unknown to exclude; label via ID_FAV -> README table (Casale ids inverted) + OpenACC spec CSVs | 14 columns: time, lead and follower position / speed / acceleration (differenced speed), gap, headway, speed difference; 10 Hz in 12 files, 5 Hz (MicroSim) and 1 Hz (CATS platoon); no steering, pedals, yaw, flags | episodes 7 s-37 min; ~3,100 non-overlapping 60-s windows (ZalaZone 1,949, AstaZero 691, Vicolungo 240, rest small) | open (figshare CC0; paper says research-only; OpenACC CC BY upstream) | 2.77 GB all; named-vehicle step3 subset 438 MB | **partial** (longitudinal-only variant; all three classes present but one car per model; ACC-driven) | **partial, good** (~1,330 BEV vs ~670 ICE windows under identical protocols; AstaZero gives matched laps) | curl from ndownloader.figshare.com by file id | ASSISTANT |
| 4 | OpenACC (EC JRC) | https://data.jrc.ec.europa.eu/dataset/9702c950-c80f-4d2f-982f-44d06ea0009f | 7 campaigns, 27 named nameplates: 6 BEV (Model 3 x2, Model S, Model X, I-Pace, e-tron, e-Golf), ICE (A8, A6, X5, A-Class, Fiat 500X, Volvo XC40, VW Polo, S-Max, Space Star, Mini, Mazda 3, Rexton), HEV / PHEV / FCEV to exclude; spec CSV per campaign (propulsion, kW, model year; no dimensions); per-sample Human / ACC flag in 5 of 7 campaigns; 20 drivers on one Fiat 500X | 10 Hz grid: GNSS speed, lat / lon, ENU position, inter-vehicle spacing; single-vehicle campaign: CAN speed, rpm, gear; no acceleration (differentiate), no steering, pedals, yaw | one continuous file per run (33 s-68 min; platoons 20.3 h = 104 vehicle-hours) + 438 single-vehicle trips (~340 h) | open, CC BY 4.0, plain HTTPS directory | 830 MB (platoon campaigns 502 MB) | **partial** (longitudinal-only; footprint from public specs; vehicle confounded with campaign -> within-campaign comparisons) | **partial** (~35 BEV vs ~36 ICE vehicle-hours, mostly ACC; + ~340 h human-driven ICE) | `wget -r` on the open directory | ASSISTANT |
| 5 | MoreTorque/rlogs (community openpilot, Mazda-centric) | https://huggingface.co/datasets/MoreTorque/rlogs | 11 platforms, 32 devices, 32,508 segments: Mazda CX-5 (2 gens), Mazda 6, CX-9, CX-30, Mazda3 (2 gens), CX-50, Hyundai Sonata 2020, Lexus RX 2016, Omoda E5 (BEV, 1 device); label = folder = carFingerprint; CX-30 / CX-50 / Mazda3 2019 / Omoda are community-fork platforms absent upstream | openpilot rlogs: carState 100 Hz (v, a, steering angle / rate / torque, pedal flags, standstill, cruise), carControl (engaged), modelV2 vision lead 20 Hz, liveLocationKalman yaw 20 Hz, liveParameters, raw CAN; no GPS, video, radarState, wall-clock | 60-s segments; 2,132 routes, 51 % with >= 5 consecutive minutes -> drive-level aggregation possible | open, Apache-2.0 | 347 GB (Omoda 8, CX-50 13, Sonata 35, CX-30 37, Mazda3 24, CX-5 80, Mazda 6 69, CX-9 59 GB) | **partial** (full channel set; SMALL and MID only, CX-9 label conflict 5.15 vs 4.87 m^2; 1-7 devices per model, so driver and car are confounded) | **sanity** (one BEV, one device, ~15 h) | `hf download --include "<folder>/*"`, staged | OWNER |
| 6 | openpilot / opendbc CI test routes | https://github.com/commaai/opendbc/blob/master/opendbc/car/tests/routes.py | 297 routes, 245 platforms, 284 devices incl. makes absent from training (BYD Atto 3, Peugeot 208, Honda e, MG5); 33 BEV routes / 27 platforms; label = opendbc CAR enum (= carFingerprint); 85 devices overlap the lab library -> 206 clean routes / 176 platforms / 14 clean BEV routes | full openpilot rlogs 2018-2025 (carState, controlsState or selfdriveState, radarState, livePose / liveLocationKalman, IMU, GPS, CAN): every channel available; schema varies by year | whole routes hosted, median 15 min, 289 / 297 unbroken, 274 >= 5 min | open anonymous HTTPS (Azure blob); licence NOT stated -> internal evaluation only, never redistribute (unmasked VINs) | 53 GB all (5,827 segments, ~97 h); clean subset 37 GB; segments 0-2 per route 8 GB; one segment per route 2.7 GB | **partial** (per-platform hit / miss, one drive per platform; ~70 platforms need spec lookups) | **partial** (14 clean BEV routes vs ~150 ICE routes; hybrids excluded by enum name) | parse routes.py -> curl loop (zst, then bz2) | OWNER for bulk; ASSISTANT for a one-segment-per-BEV-route pilot |
| 7 | CANdid (Univ. of Adelaide, VehicleSec '25) | https://doi.org/10.25909/29068553 | 10 Australian petrol cars, 7 models (Holden Astra K, Barina x2, Cruze; Mazda CX-3, CX-5 x2; Subaru Impreza x2; Toyota Yaris), all SMALL (~3.65-4.3 m^2, public); make / model / year / fuel / transmission in meta.json; none openpilot-supported | raw CAN logs decodable with opendbc DBCs: speed and wheel speeds, steering angle (+ rate), brake, accelerator, yaw rate and lat / long accel, cruise flags at 10-100 Hz; GPS 1 Hz; no radar / lead, all human-driven | one 25-65 min drive per car (6.7 h), gap-free files supplied; ~400 windows | open, CC BY 4.0 (figshare API works headless; "Download all" is browser-only) | 13.7 GB with video; logs only 4.0 GB; gap-free CAN 1.35 GB | **sanity** (10 unseen SMALL cars; ~1 day of decoding) | **sanity** (ICE only) | figshare API file list -> curl with retries (links expire in 10 s) | OWNER (4 GB logs); ASSISTANT for the 1.35 GB gap-free subset |
| 8 | A2D2 sensor-fusion bus data (Audi) | https://a2d2-dataset.github.io/ | one Audi Q7 e-tron (diesel plug-in hybrid SUV), footprint ~5.0 m^2 LARGE (public); three urban sequences, 2018-2019 | JSON bus signals: speed 50 Hz, accelerator %, brake pressure, steering-wheel angle + sign 100 Hz, IMU accel and gyro 200 Hz, pitch / roll, GPS 6 Hz; no torque, no cruise flag, no lead | 3 continuous sequences, 36.5 min -> 35 windows (29 mostly moving) | open, CC BY-ND 4.0, public S3 | 440 MB (bus JSON only; the 2.5 TB camera / lidar data is not needed) | **sanity** (single LARGE probe; low-speed urban loops) | exclude (PHEV), qualitative only | three curl calls to S3 | ASSISTANT |
| 9 | Yuan et al. 2024 Tesla Model 3 (Patterns) | https://doi.org/10.6084/m9.figshare.23904102 | one 2019 Tesla Model 3 SR+ RWD (BEV; lab footprint 4.57 m^2 MID); 106 real trips in Changchun + 46 dynamometer cycles | 20 Hz grid (speed interpolated from <= 10 Hz CAN), battery current / voltage, GPS ~1 Hz, gear; no acceleration (differentiate), no steering, pedals, lead | trips 12-50 min, gap-free; ~3,000 windows | open, CC BY 4.0 | 304 MB zip (xlsx) | **sanity** (one MID BEV, speed-only) | **sanity** (BEV-positive from a non-openpilot source, speed-only) | wget zip + md5 | ASSISTANT |
| 10 | comma2k19 (comma.ai, 2018) | https://github.com/commaai/comma2k19 | 2016 Honda Civic Touring + 2017 Toyota RAV4 Platinum, both ICE, both SMALL (4.20 / 4.13 m^2), both platforms in the training inventory; two drivers | full 2018 openpilot rlogs (carState 100 Hz incl. pedals / torque / cruise, controlsState, radarState 20 Hz), IMU 100 Hz, GNSS, processed numpy arrays; 2018 field names | 2,019 minutes in 178 routes, 119 routes with >= 10 consecutive minutes; highway I-280 only | open, MIT (Academic Torrents / HF mirror) | 94.6 GB in 10 chunk zips (video cannot be separated); two-car minimum 18 GB | **sanity** (two SMALL ICE cars; adapter and 2018-hardware check) | **sanity** (ICE negative control) | torrent or `hf download --include raw_data/Chunk_1.zip raw_data/Chunk_3.zip` | OWNER |
| 11 | UGA Mobility Lab EV platoon (OPD / TPD) | https://github.com/UGA-MOBILITY-LAB/ev-car-following-opd | 2021 Lexus UX 200 (ICE lead, ~4.09 m^2 SMALL public), 2024 Ford Mustang Mach-E (BEV, lab 4.85 MID), two 2023 Kia Niro EV (BEV, lab 4.26 SMALL); 12 scripted runs, 3 drivers; vehicle-to-folder mapping inferred, not documented | 10 Hz GNSS speed / position / altitude + 3-axis accelerometer in device frame; no steering, pedals, flags; spacing reconstructable from paired GPS | 48 gap-free files of 6-8 min (5.8 h) -> ~350 windows | open (MIT repo; Google Drive folder, anyone with the link) | 17 MB | **sanity** (GNSS-only; two classes) | **partial, weak** (BEV followers vs scripted ICE lead; OPD vs TPD factor is a regen probe) | gdown folder | ASSISTANT |
| 12 | microSIM-ACC / EV (FAU + UC Berkeley) | https://github.com/microSIM-ACC/EV | 2022 Hyundai Ioniq 5 (272 trials; lab 4.83 MID), 2023 Polestar 2 (11; lab 4.38 SMALL), 2022 Tesla Model 3 (11; lab 4.57 MID), all BEV ACC followers; human-driven Camry / Corolla leads, speed only | 50 Hz grid interpolated from 25 Hz GNSS: follower and leader speed (raw + smoothed), spacing; no steering, pedals, flags | trials 25-145 s; 110 trials >= 60 s -> ~111 windows (99 Ioniq 5) | open GitHub, no licence stated | 21 MB | **sanity** (three cars, one per class, 90 % Ioniq 5) | **sanity** (BEV-positive only; ICE companion repo at 5 Hz OBD) | git clone | ASSISTANT |
| 13 | VED - Vehicle Energy Dataset (U. Michigan) | https://github.com/gsoh/VED | 383 vehicles, Ann Arbor 2017-18: 264 ICE, 92 HEV, 24 PHEV, 3 BEV (2013 Leaf); make / model withheld; weight in 500-lb bins; Vehicle Class blank for 98 % | OBD speed 1 Hz (1 km/h steps), GPS every 3 s, engine and battery channels (drop: direct leakage); no acceleration, steering, pedals | trips ~15 min on average, trip-relative time, start / end fogged | open, Apache-2.0 | 176 MB 7z -> 2.8 GB CSV | not testable (no footprint; weight-bin ordinal check at most) | **partial, weak** (3 BEV vs 264 ICE, 1 Hz speed only; hybrids as hard negatives) | curl two archives + two xlsx | ASSISTANT |
| 14 | Ford Multi-AV Seasonal | https://avdata.ford.com/ | three 2014 Ford Fusion Hybrid AV test cars (HEV; ~4.50 m^2 MID borderline, public), fixed Michigan loop, manual driving by safety drivers | rosbag ~200 Hz: world-frame velocity, gravity-compensated IMU accel, gyro yaw rate, GPS, poses; no steering, pedals, CAN, radar | 29 bags of 8-26 min, stitchable per drive day (6.8 h) | open, CC BY-NC-SA 4.0, public S3 | 295 GB of bags (ego topics ~2 % of bytes but inseparable); one bag 5.7 GB | **sanity** (kinematics-only, one MID model) | exclude (HEV) | `aws s3 cp --no-sign-request` | OWNER (lowest priority) |

Label conflicts and corrections to carry into the adapters:

* Hyundai Ioniq 5: the lab spec table gives 2,970 x 1,625 mm = 4.83 m^2 = **MID** (0.05 below the LARGE cut); two
  verification records called it LARGE from public specs (~4.9). Use the lab value.
* Mazda CX-9: lab table 3,100 mm wheelbase -> 5.15 m^2 LARGE; published wheelbase 2,930 mm -> 4.87 m^2 MID. Report
  both or exclude the car from the size score.
* Ford Mustang Mach-E 4.85 m^2 and Audi e-tron ~4.84 m^2 (public) sit within 0.05 m^2 of the LARGE cut; Mazda 6
  (4.47) and Toyota Camry (4.47) sit within 0.05 m^2 of the SMALL / MID cut: treat as borderline.
* ULTra-AV Casale ids are inverted in its README (CSV id 0 = Hyundai Nexo, a fuel-cell EV, not an "Ioniq hybrid");
  OpenACC's Peugeot is "5008" in the spec file but "3008" in every CSV header -> exclude.
* Ohio two-vehicle "FAV 1" (Ford Fusion) in ULTra-AV was driven manually; the Ohio Tesla model is undisclosed.
* Audi Q7 e-tron (A2D2) is a diesel PHEV; Ford Fusion Hybrid (Ford AV) and Lincoln MKZ (CATS, Udacity) are hybrids:
  all outside the BEV-vs-ICE task.
* CI test routes have no stated licence and carry unmasked VINs in older logs: internal evaluation only.

## 2. Adapter plans for the top 5

Channel status words: *measured* = same quantity as training; *derived* = computed from other channels (note the
deviation); *masked* = set to missing exactly as the lab pipeline does for a dead channel group.

### 2.1 commaCarSegments

Pipeline: `database.json` -> fold legacy keys with opendbc MIGRATION, drop MOCK and fork-only keys -> remove the 464
devices shared with the lab library -> stratified selection (<= 400 segments from >= 20 devices per platform for the
53 deep platforms; every segment of the held-out and absent models) -> download -> LogReader -> per-`can` message
`CarInterface.update()` + `RadarInterface.update()` at 100 Hz (current opendbc for 2025 platforms) -> resample to 10 Hz
-> 20 channels -> exactly one window per segment.

| channels | source | status |
|---|---|---|
| `v_mps`, `a_mps2` | CarState.vEgo / aEgo (opendbc speed Kalman filter, the same code path that produced the training aEgo) | measured |
| `steer_angle_deg`, `steer_rate_degps` | CarState | measured |
| `steer_torque_norm` | CarState.steeringTorque with the lab quantile scale computed per segment (needs enough driver-steering samples inside the 60 s, else NaN -> torque group masked) | measured, weaker scale |
| `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2` | CAN yaw rate exists only on Hyundai / Kia (ESP12), VW, Ford; elsewhere yaw ~ v * tan(steer / steerRatio) / wheelbase from carParams, then curv = yaw / v, lat_acc = v * yaw | derived (steering-based, training used IMU / pose) |
| `gas_pressed`, `brake_pressed`, `steer_pressed`, `standstill` | CarState | measured |
| `das_enabled` | CarState.cruiseState.enabled OR pandaStates.controlsAllowed (10 Hz); openpilot's own enabled flag is not logged | approximate |
| `lead_valid`, `lead_d_m`, `lead_v_rel_mps`, `lead_speed_mps`, `lead_a_slope_mps2`, `thw_s`, `ttc_s` | RadarInterface tracks at 20 Hz -> nearest in-lane track as leadOne; speed = v + v_rel, slope over lead episodes, thw, ttc as in training. Vision-only platforms (Bolt, Rivian, Tesla, Ioniq 5 / 6, EV6, GV60, Leaf, VW group, Subaru, Nissan, Mazda, radarless Honda) -> lead_valid = 0, as for radarUnavailable cars in training. Hyundai / Kia / Genesis fallback: stock SCC lead distance and relative speed from SCC11 / SCC_CONTROL | measured on radar cars, masked otherwise |

Labels: platform (after MIGRATION) -> lab spec table (identical naming) -> footprint class; powertrain from the lab's
platform powertrain table (BEV vs ICE; HEV / PHEV / FCEV excluded). Report three strata: (i) device-disjoint windows of
models seen in training; (ii) device-disjoint windows of the 41 held-out models (Model Y, Leaf, GV60, Lightning, Kona
EV, Ioniq EV, Corolla, RX, ...); (iii) models absent from the lab inventory (e.g. Rivian R1 Gen1, Tesla 2025 port,
Honda e) after adding spec rows.

Caveats: same community fleet, so results are external only after de-duplication by device (461 of the 1,269 VTT
training devices appear in the index and account for 37 % of training routes); the 50-segment cap per device removes
drive-level depth (segments are independent windows; the vehicle protocol becomes a device-level mean); yaw /
curvature / lateral accel are steering-derived on most brands (check sensitivity on Hyundai / Kia / VW / Ford, where CAN
yaw is measured); the automation flag is cruise-based; the powertrain decision on BEVs runs almost entirely without lead
features.

### 2.2 commaSteeringControl

Pipeline: read each `<PLATFORM>/NNNNN.csv` straight from the zip -> pad or trim to 600 rows -> channels -> one window
per file. Format already matches (10 Hz, 60 s), so the work is the channel mask and the name mapping.

| channels | source | status |
|---|---|---|
| `v_mps`, `a_mps2` | vEgo, aEgo | measured |
| `steer_angle_deg` | steeringAngleDeg | measured |
| `steer_rate_degps` | difference of steeringAngleDeg at 10 Hz (training: CAN rate at 100 Hz, resampled) | derived |
| `steer_torque_norm` | missing (steer / steerFiltered are the controller's command, not driver torque) | masked (torque group) |
| `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2` | latAccelSteeringAngle / v, / v^2, and itself (vehicle model with roll compensation; latAccelLocalizer is the noisy measured alternative) | derived |
| `gas_pressed`, `brake_pressed` | missing | masked (pedals group) |
| `steer_pressed` | steeringPressed | measured |
| `standstill` | vEgo < 0.1 m/s (0.2 % of rows) | effectively constant 0 |
| `das_enabled` | latActive (lateral engaged; longitudinal state unknown) | approximate, ~92 % true |
| lead group | missing | lead_valid = 0 |

Three groups (lead, torque, pedals) are missing at once, while the training augmentation drops one at a time. Either
(a) train a masked variant on the lab data with exactly this mask and score it here, or (b) run the full model and
label the result "reduced input". Compare against the lab's automation-engaged windows, not the human-only protocol.

Labels: archive name -> ~45 manual renames -> lab spec table (TOYOTA_RAV4_2019 -> TOYOTA_RAV4_TSS2; Toyota / Lexus
`*_HYBRID` archives -> the merged platform's footprint, powertrain HEV -> excluded; `*_1ST_GEN` -> `*_MK1`;
MAZDA_CX_5_2022 -> MAZDA_CX5_2022); exclude SYNTHETIC_V0 and TOYOTA_MIRAI_2021; NISSAN_LEAF_2018 and its
`_Instrument_Cluster` twin are one car.

Caveats: no route, device or time ids, so overlap with the training routes cannot be excluded (optional de-duplication:
correlate each 600-sample vEgo series against the lab's raw logs); engaged-only, highway-biased, almost no standstill
or pedal activity; segments are 598-601 rows; the sampling procedure is undocumented; lateral-acceleration channels
come from the vehicle model, not the IMU.

### 2.3 ULTra-AV

Pipeline: 12 step3 CSVs -> per (source, Trajectory_ID) episode -> keep 10 Hz files (drop the 5 Hz MicroSim and 1 Hz
CATS platoon files or upsample them and flag it) -> non-overlapping 60-s windows -> channels -> labels from ID_FAV.

| channels | source | status |
|---|---|---|
| `v_mps` | Speed_FAV (GNSS Doppler, OBD, or lidar-GNSS depending on source) | measured |
| `a_mps2` | Acc_FAV (forward difference) smoothed over ~0.5 s | derived, noisier than CAN aEgo |
| `standstill` | Speed_FAV < 1 m/s (rows below 0.1 m/s were removed by the authors) | effectively 0 |
| `das_enabled` | 1 for OpenACC / MicroSim / CATS / Vanderbilt followers (ACC by protocol); 0 for Ohio SV2 (Fusion, manual) | protocol-level |
| `lead_valid` | 1 (every row has a leader) | - |
| `lead_d_m` | Spatial_Gap (bumper to bumper) | measured |
| `lead_v_rel_mps` | Speed_Diff (lead minus follower, same sign as training vRel) | measured |
| `lead_speed_mps` | Speed_LV | measured |
| `lead_a_slope_mps2` | lab slope rule over the window (or smoothed Acc_LV) | derived |
| `thw_s`, `ttc_s` | gap / v; -gap / v_rel while closing, clipped as in training | derived |
| `steer_angle_deg`, `steer_rate_degps`, `steer_torque_norm`, `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2`, `gas_pressed`, `brake_pressed`, `steer_pressed` | missing | outside the training augmentation -> longitudinal-only variant (retrain on the lab data with these nine channels removed; a zero-filled run of the full model may be reported only as a secondary number) |

Labels: ID_FAV -> model per the README table of each source (Casale: id 0 = Hyundai Nexo, id 1 = SsangYong Rexton).
Footprint from the lab table where the model exists: Tesla Model 3 4.57 MID, Model S 4.97 LARGE, Model X 5.06 LARGE,
Toyota Corolla 4.09 SMALL, Hyundai Nexo 4.52 MID, Audi A4 4.37 SMALL; otherwise public specs: Audi e-tron ~4.84 MID
(borderline), Jaguar I-Pace ~4.93 LARGE, BMW X5 ~5.0 LARGE, Mercedes GLE 450 ~5.04 LARGE, Audi A6 ~4.75 MID, Ford S-Max
~4.58 MID, Peugeot 5008 ~4.56 MID, Mercedes A-Class ~4.25 SMALL, Mazda 3 ~4.29 SMALL, VW e-Golf ~4.04 SMALL. Powertrain
from the OpenACC spec CSVs: BEV = Model 3 / S / X, e-tron, I-Pace, e-Golf; ICE = A6, X5, A-Class, Mazda 3, S-Max,
Peugeot 5008, Corolla; exclude A4 Avant and GLE 450 (48-V mild hybrids, listed HEV), BMW i3s REx, RAV4 HEV, Kia Niro
PHEV, Outlander PHEV, Nexo FCEV, Lincoln MKZ, Ford Fusion, the unnamed Vanderbilt SUV and Ohio Tesla, Waymo / Argoverse.

Caveats: the longitudinal profile is the ACC controller's, not a driver's; one physical car per model (report per model,
bootstrap over windows within a campaign); AstaZero is the cleanest matched comparison (Model 3 vs A6 / X5 / A-Class on
identical laps), ZalaZone is BEV-heavy at 30-60 km/h; Casale / Vicolungo 10 Hz is a spline of 3-5 Hz GNSS; no
standstill; cite ULTra-AV and the source campaigns (OpenACC CC BY) and keep the use research-only.

### 2.4 OpenACC (raw files)

Same longitudinal-only channel map as 2.3, built from the raw campaign CSVs (5 metadata rows, then the header; column
suffix n = n-th vehicle in Vehicle_order): `v` = SpeedN; `a` = smoothed derivative; for follower n+1 the gap is IVSn and
`v_rel` = SpeedN - SpeedN+1; `das_enabled` = DriverN == "ACC" per sample (AstaZero, Cherasco, Vicolungo, Casale, JRC
low speed) or the file-level ACC flag (ZalaZone); leaders have no lead -> lead_valid = 0. Curvature from GNSS heading is
usable only at AstaZero (2-cm RT-Range) and should be left missing everywhere for consistency. The raw files add what
ULTra-AV dropped: Cherasco (Volvo XC40 ICE and Hyundai Ioniq HEV followers; Fiat 500X and VW Polo leaders), JRC low
speed (Nexo, S-Max, an unnamed BMW), and the one-vehicle campaign (Fiat 500X diesel, 20 human drivers, ~340 h of CAN
speed; drop engine rpm and gear, which trivially reveal the engine). Labels: Vehicle_order header -> spec CSV
(propulsion type, model year) -> footprint from public specs (Audi A8 ~4.91 LARGE, Volvo XC40 ~4.36 SMALL, Fiat 500X
~3.97, VW Polo ~3.61, Mitsubishi Space Star ~3.49, Mini Cooper ~3.75, SsangYong Rexton ~4.70 MID). Exclude the Peugeot
(3008 vs 5008 unresolved) and the unnamed BMW / Smart / Skoda leaders. Caveats as in 2.3, plus per-column NaN QC
(vehicles drop out of the platoon), occasional negative spacings, and no Driver column in ZalaZone.

### 2.5 MoreTorque/rlogs

Pipeline: `hf download` per folder -> LogReader with the pinned 2025 schema (log.capnp from openpilot commit a8fd557,
car.capnp from opendbc commit aa08e49) -> the lab's drive builder (carState-indexed, 10 Hz) with three substitutions
below -> windows and drives per route.

| channels | source | status |
|---|---|---|
| `v_mps`, `a_mps2`, `steer_angle_deg`, `steer_rate_degps`, `gas_pressed`, `brake_pressed`, `steer_pressed`, `standstill` | carState 100 Hz | measured |
| `steer_torque_norm` | carState.steeringTorque with the lab per-drive scale | measured |
| `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2` | liveLocationKalman(DEPRECATED).angularVelocityCalibrated[2] at 20 Hz (carState.yawRate is 0 on all Mazdas, populated on the Sonata); curv = yaw / v, lat_acc = v * yaw as in training | measured |
| `das_enabled` | carControl.enabled OR carState.cruiseState.enabled (controlsState is not retained) | measured; fork semantics on CX-30 / CX-50 / Mazda3 2019 / Omoda to be checked |
| lead group | modelV2.leadsV3[0] at 20 Hz: lead_valid from the lab's vision-lead probability rule, d = x, v_rel = v_lead - vEgo, lead_speed = v_lead, slope / thw / ttc as in training, lead_source = vision (training already holds vision leads on radarUnavailable cars); optional radar lead for the Sonata by decoding SCC11 from raw CAN | measured (vision) |

Labels: folder -> platform -> lab spec table: CX-5 and CX-5 2022 = 4.31 SMALL; Mazda 6 = 4.47 MID (0.04 above the cut);
Sonata = 4.62 MID; Lexus RX = 4.55 MID (a held-out model of the headline split); CX-9 = 5.15 LARGE by the lab table but
4.87 MID by published dimensions -> report both or exclude. Not in the table (public specs): CX-30 ~4.16 SMALL, CX-50
~4.67 MID, Mazda3 BP ~4.28 SMALL, Mazda3 BM ~4.21 SMALL, Omoda E5 ~4.08 SMALL. Powertrain: Omoda E5 = BEV; all others
ICE (the Sonata and RX fingerprints are the non-hybrid variants; Mazda 24-V mild hybrids follow the lab's mild-hybrid
convention).

Caveats: 1-7 devices per model, so driver and car are confounded: aggregate per route and per device and report per
model, never pooled; Mazda supplies ~90 % of the hours; four platforms come from a community fork (print
carParams.carFingerprint and check the cruise / engaged semantics on one segment each before scoring); 97 % of the data
were recorded July-September 2024 and the Omoda routes are undated; no GPS or wall-clock; one BEV from one device is an
anecdote, not a powertrain test.

### Model variants to prepare (summary)

* V0, as trained (20 channels): commaCarSegments, MoreTorque, CI routes, comma2k19, CANdid (lead masked), A2D2 (lead
  and torque masked).
* V1, lateral + speed mask (lead, torque, pedals off): commaSteeringControl.
* V2, longitudinal-only (speed, accel, standstill, automation flag, lead group): ULTra-AV, OpenACC, UGA, microSIM.
* V3, speed-only (speed, accel, standstill): Yuan Tesla, VED, OpenACC one-vehicle campaign.

## 3. Long tail (unverified, one line each)

Items of the scout's long tail that were promoted to the verified table (commaSteeringControl, OpenACC, A2D2, comma2k19,
VED) are omitted here.

* **UAH-DriveSet** (Univ. of Alcala) - http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/ - free registration - six known cars from city cars (Kia Picanto, Citroen C-Zero BEV) to an Audi Q5; GPS speed 1 Hz, phone-IMU acceleration, camera lead distance / TTC, no steering or pedals; instructed aggressive / drowsy driving; size medium-low (1 MID vs 5 SMALL), powertrain medium-low (one BEV).
* **Argonne D3 Downloadable Dynamometer Database** - https://www.anl.gov/taps/downloadable-dynamometer-database - open - named BEV / PHEV / HEV / ICE models on standard cycles at 10 Hz with speed, acceleration and pedals; prescribed speed trace, no lateral channels; powertrain medium (controlled), size low.
* **TUM BMW i3 battery and heating data in real driving cycles** - https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles - free registration (IEEE DataPort) - 72 trips at native 10 Hz with speed, acceleration, throttle, regen flag; one SMALL BEV; strong BEV-positive probe.
* **KU Leuven BEV Energy Dynamics (Tesla Model Y SR / LR)** - https://doi.org/10.48804/8KPDTW - open - ~1,400 real drives, speed and pack current; single MID BEV; BEV-positive, speed-only.
* **CATS Lab Tesla traffic-light / stop-sign control dataset (Model Y, 10 Hz)** - https://github.com/CATS-Lab/Data-Tesla-Traffic-Light-and-Stop-Sign-Control - open - urban L2 longitudinal automation; single MID BEV.
* **FHWA Central Ohio driver-assistance (L2) single- and two-vehicle data** - https://data.transportation.gov/d/iie8-uenj and https://data.transportation.gov/d/vhz2-exyi - open - Tesla sedan with L2 engaged (~112 h) and Ford Fusion at L0 (24 h), lidar / GNSS-derived kinematics, no steering or pedals; both MID; the raw source of ULTra-AV's Ohio files.
* **FHWA CARMA-1 CACC platooning (five 2013 Cadillac SRX)** - https://data.transportation.gov/d/wpek-zziu - open - CAN-level steering-wheel angle, pedals, yaw and radar range; single MID ICE model; followers under research CACC (use the mode flags).
* **Vanderbilt / Univ. of Arizona ACC datasets** - https://acc-dataset.github.io/datasets/ - open - seven 2018 ACC vehicles plus an 8-car platoon, make / model withheld (body style only); weak labels, six ICE + one hybrid.
* **CATS Lab Lincoln MKZ ACC and AV-platooning data** - https://github.com/CATS-Lab/Filed-Experiment-Data-ACC_Data - open - one MID research AV, powertrain variant unstated, unlabeled human vehicles; tiny.
* **MicroSimACC / ICE (2020 Toyota Corolla LE)** - https://github.com/microSIM-ACC/ICE - open - 5 Hz OBD speed / rpm / MAF plus spacing; matched ICE control for the EV repo (same lab, protocol and roads); single SMALL.
* **HISTORIC car-following platoon data (Harbin)** - https://github.com/CATS-Lab-USF/HISTORIC-data - open - speed-only, one mid-size ICE model; Chinese-road sanity check only.
* **comma.ai 2016 research driving dataset** - https://archive.org/details/comma-dataset - open - one SMALL combustion sedan (2016 Acura ILX), 7.25 h highway, pre-openpilot log format with speed, steering angle, gas and brake but no cruise flag; adapter robustness probe.
* **ANAID - naturalistic obstacle-avoidance dataset** (Univ. Europea de Madrid) - https://zenodo.org/records/18845673 - open - one MID hybrid SUV in urban evasive manoeuvres; domain-shift probe; hybrid, exclude for powertrain.
* **sunnypilotCarSegments** - https://huggingface.co/datasets/sunnypilot/sunnypilotCarSegments - open - emerging fork counterpart of commaCarSegments; negligible today, re-check in a few months.
* **HCRL / OCSLAB Kia Soul driving dataset** (Korea Univ.) - https://sites.google.com/hksecurity.net/hcrl/Datasets/driving-dataset - open - 10 drivers, 51 OBD signals incl. steering angle / rate, brake pressure, accelerator, lateral / longitudinal acceleration at 1 Hz; single SMALL ICE.
* **X-CANIDS in-vehicle signal dataset (2017 Hyundai LF Sonata)** - https://ieee-dataport.org/open-access/x-canids-dataset-vehicle-signal-dataset - free registration - fully decoded CAN at >= 10 Hz (steering, yaw, pedals, wheel speeds); single MID ICE.
* **TOTEMS 2010 Toyota Camry hybrid vs conventional** (Univ. of Vermont, Dryad) - https://doi.org/10.5061/dryad.2bvq83bnj - open - same body with HEV vs ICE powertrain at 1 Hz; probe for false-BEV calls on strong hybrids.
* **ZTBus - Zurich transit bus dataset** (ETH) - https://doi.org/10.3929/ethz-b-000626723 - open - two 19-m trolleybuses with steering angle, brake pressure and traction power; out-of-distribution LARGE / electric probe only.
* **DDD20 / DDD17 DAVIS driving dataset** - https://sites.google.com/view/davis-driving-dataset-2020/home - open - Ford Mondeo (MID) and Ford Focus (SMALL) at exactly 10 Hz with steering-wheel angle, speed, accelerator % and brake flag; combustion only; a clean two-class size probe.
* **nuPlan** (Motional) - https://www.nuscenes.org/nuplan - free registration - Chrysler Pacifica Hybrid (LARGE PHEV) with ego kinematics and agent tracks (lead derivable), no steering or pedals; long logs; powertrain ambiguous.
* **KITTI raw / KITTI-360** - https://www.cvlibs.net/datasets/kitti/raw_data.php - free registration - VW Passat wagon (SMALL, ~4.2 m^2) with OXTS speed / acceleration / yaw rate at 10 Hz, no steering or pedals; ~1-1.5 h of drives longer than 60 s; ICE.
* **Zenseact Open Dataset (ZOD) Drives** - https://zod.zenseact.com/drives/ - data request / agreement - steering, pedals, speed and IMU at 100 Hz but the vehicle model is undisclosed; consistency checks only.
* **nuScenes + CAN bus expansion** - https://www.nuscenes.org/nuscenes - free registration - Renault Zoe (SMALL BEV) with near-complete CAN, but 20-s scenes are too short for the 60-s protocol.
* **Oxford RobotCar / Radar RobotCar** - https://robotcar-dataset.robots.ox.ac.uk/ - availability unknown (site unreachable at check) - Nissan Leaf (SMALL BEV) with 50 Hz INS; high value if access returns.
* **Udacity self-driving car datasets CH2 / CH3 (Lincoln MKZ)** - https://github.com/udacity/self-driving-car/tree/master/datasets - open - steering, throttle, brake, speed at ~50 Hz plus IMU; one MID hybrid; combustion with caveat.
* **Boreas** (Univ. of Toronto) - https://www.boreas.utias.utoronto.ca/ - open (AWS) - Buick of undisclosed model, 200 Hz kinematics; consistency only.
* **HDD - Honda Research Institute Driving Dataset** - https://usa.honda-ri.com/HDD - data request - 104 h with CAN steering, pedals and speed from one vehicle of unconfirmed identity; consistency only.
* **SHRP 2 Naturalistic Driving Study** (VTTI InSight) - https://insight.shrp2nds.us/ - data-use licence (IRB training, fees, custom extraction) - thousands of named 2010-2013 vehicles at 10 Hz with steering, pedals and radar; the best potential size test and the hardest access; almost no BEV.
* **Safety Pilot Model Deployment (SPMD)** (UMTRI) - https://data.transportation.gov/d/deww-aa3r - open - 10 Hz naturalistic data with radar range but no make / model; unlabeled distribution check only.
* **UDRIVE** (EU FP7) - https://www.udrive.eu/ - data request through consortium partners, continued availability unconfirmed - three small Renault models, combustion only.
* **PIE - Pedestrian Intention Estimation** (York Univ.) - https://data.nvision2.eecs.yorku.ca/PIE_dataset/ - open - vehicle unknown, speed and heading only; poor.
* **MIT-AVT** - https://avt.mit.edu/ - not public (industry consortium) - would be excellent (Tesla S / X / 3 and Mach-E vs Volvo S90, Range Rover Evoque, Cadillac CT6 with L2 engaged) but no release process exists.

## 4. Needs manual download by the owner: exact steps

Set once: `export EXT=<folder for external data on the large data disk>` and `pip install -U huggingface_hub` (gives the
`hf` command; the older `huggingface-cli download` accepts the same flags). Optional `hf auth login` with a free token
avoids anonymous rate limits when pulling tens of thousands of files. Downloaded archives are untrusted data: keep them
in their own empty directories, separate from code.

### 4.1 commaCarSegments (first priority)

1. Index (9.4 MB): `curl -L --create-dirs -o $EXT/ccs/database.json https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main/database.json`
2. Selection list: the assistant builds `$EXT/ccs/select.txt` (one `<dongle>/<route>/<seg>` per line) from the index:
   legacy keys folded with opendbc MIGRATION, MOCK and fork keys dropped, the 464 lab-overlapping dongles removed,
   <= 400 segments from >= 20 devices per platform for the 53 deep platforms plus every segment of the held-out and
   absent models (~20,000 segments, ~32 GB).
3. Download (resumable, 8 parallel):
   `xargs -P 8 -I{} sh -c 'curl -sfL -C - --create-dirs -o "$EXT/ccs/segments/{}/rlog.zst" "https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main/segments/{}/rlog.zst"' < $EXT/ccs/select.txt`
   (equivalent: `hf download commaai/commaCarSegments --repo-type dataset --local-dir $EXT/ccs --include 'segments/<dongle>/<route>/<seg>/rlog.zst' ...`, or `huggingface_hub.snapshot_download(allow_patterns=[...])` with the list).
4. Budget: ~1.6 GB per 1,000 segments; the full set (299 GB) is not needed.
5. Optional helper: `pip install https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main/dist/comma_car_segments-0.1.0-py3-none-any.whl` (needs opendbc) for `get_comma_car_segments_database()` and `get_url()`.

### 4.2 commaSteeringControl

1. All nine BEV archives (3.3 GB):
   `hf download commaai/commaSteeringControl --repo-type dataset --local-dir $EXT/csc --include 'data/KIA_EV6_2022.zip' 'data/CHEVROLET_BOLT_EUV_2022.zip' 'data/HYUNDAI_IONIQ_5_2022.zip' 'data/KIA_NIRO_EV_2020.zip' 'data/NISSAN_LEAF_2018.zip' 'data/NISSAN_LEAF_2018_Instrument_Cluster.zip' 'data/HYUNDAI_KONA_ELECTRIC_2019.zip' 'data/GENESIS_GV60_ELECTRIC_1ST_GEN.zip' 'data/HYUNDAI_IONIQ_ELECTRIC_LIMITED_2019.zip'`
2. ICE archives chosen by footprint class from the lab table (for example SMALL: HONDA_CIVIC_2016, TOYOTA_COROLLA_TSS2_2019; MID: HYUNDAI_SONATA_2020, HONDA_ACCORD_2018, TOYOTA_CAMRY; LARGE: RAM_1500_5TH_GEN and the Odyssey / Silverado / Sienna archives present in the file tree): same command with those `data/<PLATFORM>.zip` names; ~1-1.4 GB each.
3. Everything: `--include 'data/*.zip'` (45.5 GB; skip SYNTHETIC_V0.zip). Read the CSVs directly from the zips (zipfile + pandas) rather than unzipping (3x expansion).
4. Single-file URL form: `https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/<PLATFORM>.zip`.

### 4.3 MoreTorque/rlogs (staged)

1. `hf download MoreTorque/rlogs --repo-type dataset --local-dir $EXT/moretorque --include "OMODA_E5/*"` (8 GB, the only BEV).
2. Then `--include "MAZDA CX-50/*"` (13 GB), `"HYUNDAI SONATA 2020/*"` (35 GB), `"MAZDA CX-30/*"` (37 GB), `"MAZDA 3 2019/*"` (24 GB) for unseen-platform and new-device tests; `"MAZDA CX-9/*"` (59 GB), `"MAZDA 6/*"` (69 GB), `"MAZDA CX-5/*"` (80 GB), `"MAZDA CX-5 2022/*"` (22 GB) if disk allows; omit `--include` for all 347 GB. Folder names contain spaces: keep the quotes.
3. First check after download: print carParams.carFingerprint from one Omoda E5 and one CX-50 segment.

### 4.4 openpilot / opendbc CI test routes (internal evaluation only)

1. Index (297 lines, route and platform):
   `curl -sL https://raw.githubusercontent.com/commaai/opendbc/master/opendbc/car/tests/routes.py | grep -oE 'CarTestRoute\("[^"]+",\s*[A-Z_]+\.[A-Z0-9_]+' | sed -E 's/CarTestRoute\("([^"]+)",\s*/\1\t/' > $EXT/ci/ci_routes.tsv`
2. Fetch (try zst, then bz2; 13 routes have gaps or start late, so stop only after 8 consecutive misses):
   `B=https://commadataci.blob.core.windows.net/openpilotci; while IFS=$'\t' read -r R P; do s=0; miss=0; while [ $miss -lt 8 ]; do got=0; for f in rlog.zst rlog.bz2; do curl -sfL --create-dirs -o "$EXT/ci/$P/${R//\//__}/$s/$f" "$B/$R/$s/$f" && { got=1; break; }; done; [ $got = 1 ] && miss=0 || miss=$((miss+1)); s=$((s+1)); done; done < $EXT/ci/ci_routes.tsv`
3. Variants: one segment per route ~2.7 GB; segments 0-2 ~8 GB; everything ~53 GB; the assistant supplies the list of the 206 routes whose devices are not in the lab library (clean subset ~37 GB).
4. No licence is stated and older logs carry unmasked VINs: evaluate internally, do not redistribute, ask comma (GitHub Discussions / Discord) before publishing derived data.

### 4.5 comma2k19

1. Two-car minimum (18 GB): `hf download commaai/comma2k19 --repo-type dataset --local-dir $EXT/comma2k19 --include "raw_data/Chunk_1.zip" "raw_data/Chunk_3.zip"` (Chunk_1 = RAV4, Chunk_3 = Civic); everything: `--include "raw_data/*"` (94.6 GB). Torrent alternative: `aria2c --seed-time=0 https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent` and select the chunk zips.
2. Unzip on ext4 / xfs (route names contain a vertical bar); on NTFS / exFAT use the repo's `utils/unzip_msft_fs.py`.
3. Zero-cost smoke test first: `git clone https://github.com/commaai/comma2k19` and point the adapter at `Example_1/`.

### 4.6 CANdid

1. File list without video (~4 GB of logs):
   `curl -s https://api.figshare.com/v2/articles/29068553 | python3 -c 'import json,sys; [print(f["download_url"], f["name"]) for f in json.load(sys.stdin)["files"] if not f["name"].endswith(".mp4")]' > $EXT/candid/files.txt`
2. Download with retries (pre-signed links expire after 10 s; an XML body means "expired", so retry):
   `cd $EXT/candid && while read url name; do for i in 1 2 3; do curl -sSL -o "$name" "$url" && ! head -c 5 "$name" | grep -q '<?xml' && break; sleep 1; done; done < files.txt`
3. Browser alternative: open https://adelaide.figshare.com/articles/dataset/CANdid/29068553 and use "Download all" (13.7 GB zip incl. video; the page refuses non-browser clients).
4. Decoding afterwards: `pip install cantools`; DBCs from github.com/commaai/opendbc (Toyota 2017, GM Global A powertrain, Mazda 2017, Subaru pre-global 2015); resample to 10 Hz.

### 4.7 Ford Multi-AV Seasonal (optional, lowest priority)

1. `pip install awscli`; list with `aws s3 ls --no-sign-request --recursive s3://ford-multi-av-seasonal/`.
2. One bag probe (5.7 GB): `curl -O https://ford-multi-av-seasonal.s3-us-west-2.amazonaws.com/2017-10-26/V2/Log5/2017-10-26-V2-Log5.bag`; all 29 bags (295 GB): `aws s3 cp --no-sign-request --recursive --exclude '*' --include '*.bag' s3://ford-multi-av-seasonal/ $EXT/ford_av/`.
3. Parse without ROS: `pip install rosbags` (topics /imu, /velocity_raw, /gps, /gps_time, /pose_raw); delete each bag after extraction. Only worth it if disk is free, since the result is a single-model kinematics-only sanity check.

## 5. Assistant-fetchable (open, small): can be scripted in-session on request

| dataset | size | command |
|---|---|---|
| ULTra-AV named-vehicle step3 subset (12 files) | 438 MB | `curl -L -o <name> https://ndownloader.figshare.com/files/<id>` for ids 59770103 (ZalaZone), 59770094 (AstaZero), 59770088 (Vicolungo), 59770070 (Casale), 59770073 (MicroSim, 5 Hz), 59770082 (CATS ACC), 59770076 (CATS platoon, 1 Hz), 59770079 (CATS UWM), 59770067 (Vanderbilt), 59770091 (Ohio single), 59770085 (Ohio two), 59770010 (README); ids change with the figshare version, re-list with `curl -s https://api.figshare.com/v2/articles/26339512` |
| OpenACC (all 7 campaigns) | 830 MB | `wget -r -np -nH --cut-dirs=4 -R 'index.html*' 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/'` (add `-X '*One_vehicle*'` for the 502 MB platoon campaigns only) |
| A2D2 bus-signal JSONs (3 sequences) | 440 MB | `curl -O https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar/<seq>/bus/<seq-compact>_bus_signals.json` for 20180810_150607, 20190401_121727, 20190401_145936 |
| Yuan et al. 2024 Tesla Model 3 | 304 MB | `wget -O DATA_S1_update20241031.zip https://ndownloader.figshare.com/files/50103315` (md5 628a3f0347662de30d958ada5d611f3e); Table S3 trip list from the article's supplementary mmc2.xlsx |
| VED | 176 MB | `git clone --depth 1 https://github.com/gsoh/VED.git` or curl the two 7z archives and two xlsx under `Data/`; `7z x` both archives |
| microSIM-ACC / EV | 21 MB | `git clone https://github.com/microSIM-ACC/EV.git`; `unzip` the three zips into empty folders; optional ICE companion `git clone https://github.com/microSIM-ACC/ICE.git` |
| UGA EV platoon | 17 MB | `pip install gdown && gdown --folder 'https://drive.google.com/drive/folders/18L-EAKB0W2g7jJbiFH8zRxq0K0Toa28y'` |
| CANdid gap-free CAN subset + GPS / meta (pilot) | 1.35 GB | the API listing of 4.6 filtered to `*_driving_CAN_continuous.log`, `*.gps`, `*.json`, README |
| commaSteeringControl single platform zips (pilot) | 26 MB-1.4 GB each | `hf download commaai/commaSteeringControl --repo-type dataset --include 'data/<PLATFORM>.zip' --local-dir <dir>` |
| CI routes, one segment per BEV route (pilot) | ~0.3 GB | first segment of the 33 BEV routes from the fetch loop of 4.4 |

Verification copies of the small sets (ULTra-AV step3 subset, OpenACC spec and notes files, the three A2D2 bus JSONs,
microSIM EV, CANdid samples and metadata, the commaCarSegments index) already exist in the project scratch area from the
verification pass and can be reused instead of fetching again.

## 6. Suggested order of work

1. In-session, small sets: build the longitudinal-only variant (V2) and score ULTra-AV (AstaZero matched laps first,
   then ZalaZone) and raw OpenACC; run the sanity probes A2D2 (LARGE), Yuan Tesla (MID BEV, V3), UGA and microSIM (V2),
   VED (V3, weight-bin ordinal check).
2. Owner downloads, in this order: commaCarSegments selection (~32 GB, the only full-protocol external test),
   commaSteeringControl BEV + ~10 ICE archives (~8 GB, V1), MoreTorque Omoda + CX-50 + Sonata (~56 GB, V0).
3. Optional: CI routes clean subset (37 GB, internal only), CANdid logs (4 GB plus a day of decoding), comma2k19 two
   chunks (18 GB, adapter and hardware-era check). Skip Ford Multi-AV unless disk is free.
