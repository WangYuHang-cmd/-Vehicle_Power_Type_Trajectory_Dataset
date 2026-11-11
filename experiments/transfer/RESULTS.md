# Zero-shot transfer of the VTT models to external datasets: results

Written 9 Nov 2026 by the scorer step of the transfer campaign (`experiments/transfer/run_scoring.sh` ->
`score.py`; tables collected by `collect_scores.py` into `RESULTS_TABLES.md`, in-distribution references by
`internal_reference.py` / `internal_per_model.py`, one diagnostic by `diag_derived_accel.py`). Nothing was fine-tuned:
the final `split_platform` checkpoints (size: bi-GRU S9, 3 seeds; BEV vs ICE: 1-D ResNet B5, 3 seeds) and the
channel-subset variants of `VARIANTS.md` (V1 lateral + speed, V2 longitudinal-only, V3 speed-only; 2 seeds each, same
recipes) were run as they are on the window sets built by the adapters (`DS_*.md`, contract `CONTRACT.md`). All
numbers are seed ensembles (mean of the seeds' window probabilities), aggregated by the lab rule: window -> drive
(mean probability vector of the drive's windows -> arg-max) -> vehicle (all drives of one physical car pooled) -> car
model (all windows of the model). Accuracy and macro-F1 in percent with one decimal; macro-F1 only where at least two
classes are present; 95 % intervals from a cluster bootstrap over vehicles (1,000 resamples), which is uninformative
for sets with one or two cars. Chance = 1 / K (50.0 / 33.3) and the majority share of the set. "Internal reference" =
the same seed ensemble on our own `split_platform` test split (41 held-out car models), overall and on the matching
sub-population (`das` = windows with the driving automation engaged on >= 50 % of the samples, for the ACC-driven
sets; `human` = strict human-only windows, for the manually driven sets). External vehicle ids never appear here; the
car names are the public labels of the datasets. Sections 2.10-2.11 and the phase-2 parts of sections 0, 3, 4 and 5 were
added later on 9 Nov by the phase-2 scorer step, after the owner's downloads (commaSteeringControl, MoreTorque) had been
turned into window sets by their adapters (`DS_CSC.md`, `DS_MORETORQUE.md`); the phase-1 text and numbers are unchanged.
Section 2.12 and the phase-3 parts of sections 0, 1, 3, 4 and 5 were added on 10 Nov by the phase-3 writer step, after the
owner's device-disjoint commaCarSegments selection (`CCS_SELECTION.md`) had been downloaded, decoded by `ccs_decode.py`
(`CCS_PIPELINE.md`) and scored with the nine runs of the `ccs` case of `run_scoring.sh`; the phase-1 and phase-2 text is unchanged
except where a pooled total now includes the new set. Sections 2.13-2.14 and the matching parts of sections 0, 1, 3, 4 and 5
were added on 10 Nov (evening) by the phase-3b writer step, after the nine Honda platforms the phase-3 decode had lost were
re-decoded with a gearbox option of `ccs_decode.py` (`ccs_honda`) and the two comma2k19 chunks were decoded by the same decoder
(`comma2k19`); the earlier text is unchanged except where a pooled total now includes the two sets and where 2.12 points to 2.13.
Section 2.15, the understeer block of section 0, three rows of 1.1, one rule of 1.3, 3.7 and the matching parts of 2.12-2.14, 4 and 5
were added on 11 Nov by the understeer writer step, after the owner had approved the understeer-corrected re-decode of the three
steering-derived sets (`ccs_us`, `ccs_honda_us`, `comma2k19_us`; `UNDERSTEER.md`, `CCS_PIPELINE.md` "Understeer-corrected yaw");
the as-run numbers of 2.12-2.14 and every pooled total are unchanged, and 2.14, 3.6 point 9 and section 4 were reworded where they
called the comma2k19 `yawRate` a measurement (it is the output of 2018 openpilot's vehicle model, `UNDERSTEER.md` section 9).

## 0. Headline

Phase 3 (one set with all 20 channels, the car's own CAN acceleration and device ids, on 4,061 devices the lab never saw;
2.12, 3.6; the two sets decoded afterwards, 2.13-2.14, follow in the next block):

* **Powertrain reproduces the lab's held-out test at model level on a fleet of 4,061 unknown devices, and the held-out
  stratum reproduces it at device level.** commaCarSegments, device-disjoint selection: 45,763 one-minute windows
  (762.7 h) of 113 car models on 161 openpilot platforms, 39.3 % of the minutes automation-engaged. The full 20-channel
  model calls 86 of 88 car models right (97.7 % [96.2, 100.0]; BEV 14/15, ICE 72/73; the misses are the Honda e with 2
  windows and one engaged Escalade window), 87.8 % [86.6, 89.1] of 2,512 devices (macro-F1 84.7; internal 93.9 / 93.4)
  and 81.8 % of the windows (internal 82.5). On the 29 held-out test models of `split_platform` (the external replica of
  the protocol) it is 27 of 27 labelled models and 94.2 % [92.0, 96.2] of 463 devices; on the training / validation
  models on new devices (51 of the 70 carry a BEV / ICE label) 50 of 51 models but 86.1 % of 2,041 devices: a seen platform on a new device is not easier than a
  new platform. The Nissan Leaf, which every speed-only model misses (VED, internal V3, V3 here), is BEV for 22 of 28
  devices with the full model.
* **The device-level loss is the engaged minutes, and part of it is the lead group.** Window accuracy 92.2 % on the
  27,759 human-driven minutes, 64.8 % on the 18,004 engaged ones (P(BEV) on ICE windows 0.12 against 0.39); the same
  devices are 93.8 % right from their human minutes and 70.9 % from their engaged minutes (internally the full model
  loses 0.7 points on the engaged subset). Masking the lead group, which here is a radar in-path gate or the stock-ACC
  object rather than radard's fused lead, lifts the engaged-minute device accuracy to 81.8 % and the overall one to
  91.3 %; the speed-only variant V3, without a lead, is the best powertrain model on this fleet (92.6 % [91.5, 93.6] of
  devices, 87 of 88 models; internal 91.0) and V2 with the pedal flags but no lateral channels is not better than V3
  (84.6 %), so the pedal flags, the reason the set was downloaded, add nothing measurable.
* **Size: SMALL transfers (49 of 50 models, 91.1 % of 1,762 devices), MID and LARGE do not (8 of 33 and 6 of 30 models,
  27.6 % and 23.5 % of their devices), and the seen stratum is no better than the held-out one.** Full model 54.1 %
  [52.7, 55.6] of devices / macro-F1 46.4 (internal 63.5 / 59.7), 63 of 113 models; seen 53.9 % of 3,199 devices and 40 of
  70 models, held-out 53.8 % of 864 devices and 13 of 29 models (internally 19 of the same 29). The MoreTorque Sonata
  generalises: a training platform on a new device is not in distribution for the size model. LARGE survives for pickups
  only (Silverado, RAM 1500 / HD, F-150, F-150 Lightning); every minivan and three-row SUV is SMALL (Odyssey 2 of 96
  devices, Pilot 2/77, Sienna 15/94, Carnival 0/15, Atlas 0/32, Pacifica 27/184), including the Carnival and Sienna that
  the internal test got right; 6 of the 8 MID hits are BEV platforms (Tesla, E-GMP, Mach-E). The misses are systematic per
  platform, so more minutes per device lower the size accuracy (60.0 % with one window, 50.3 % with 10 or more) while they
  raise the powertrain accuracy (80.1 -> 96.3 %). The speed-only V3 reaches its internal size number (53.7 % of devices /
  macro-F1 50.0, 14 of 30 LARGE models from the `aEgo` texture) and beats the full model's macro-F1; the lateral channels
  carry the full model's decision (masking the yaw group calls 69.6 % of the devices LARGE, 88.3 % of the LARGE ones) but, steering-derived on a
  new fleet, they do not deliver the nine points they add internally.
* **Pooled over all sets** (3.1, 3.2; one physical car, archive or device x model = one decision; the totals include the
  two sets of the next block): BEV 575 / 667 (86.2 %), ICE 2,224 / 2,472 (90.0 %), all 2,799 / 3,139 (89.2 %); size SMALL
  1,912 / 2,084 (91.7 %), MID 366 / 1,338 (27.4 %), LARGE 254 / 1,072 (23.7 %), all 2,532 / 4,494 (56.3 %); the
  commaCarSegments devices (`ccs` + `ccs_honda`) contribute 2,778 and 4,384 of these decisions (`ccs` alone 2,547 and 4,152:
  powertrain 2,232 / 2,547, size 2,245 / 4,152).

Phase 3, two sets decoded after the write-up above (2.13, 2.14; 3.6 points 7-9):

* **The nine Honda platforms the phase-3 decode had lost are recovered and read as SMALL and ICE (`ccs_honda`, 2.13).**
  The loss was the parser, not the data: opendbc 0.3.1 waited for a gearbox message these traditional-CVT Hondas never
  send; a decoder option that takes the gearbox type from the CAN addresses (byte-identical otherwise, cross-checked
  against the lab's 2024-08 openpilot stack) recovers 2,189 of the 2,198 segments -> 2,132 windows (35.5 h) from 230
  devices on 9 platforms (Civic 2016 / Bosch / 2022, CR-V 2016 / 5G, Fit, HR-V 2016 / 3G, Freed). Full model: SMALL for
  every platform, 226 of 230 devices (98.3 % [96.5, 99.6]) and 5 of 5 models; ICE for every labelled platform, 227 of 229
  devices (99.1 % [97.8, 100.0]) and 4 of 4 models; the held-out HR-V 5 of 5 on both tasks (P(SMALL) 0.73, P(ICE) 0.84,
  internally 46.0 % drive accuracy / SMALL 0.50 and 95.0 % / ICE 0.84). Both HONDA_CIVIC_2022 misses are among the 9
  devices whose gearbox message says `automatic`, most likely the Civic Hybrid filed under the same name (without them 219
  of 220 devices); engaged minutes score lower (window accuracy 73.7 against 91.1 % size, 80.7 against 95.6 % powertrain);
  the yaw mask collapses the size call to 14.3 % of the devices, the lead mask changes nothing; V3 86.5 / 98.3 %, V2 73.0 /
  97.8 %, V1 47.8 / 86.0 % of the devices (size / powertrain).
* **Two cars on 2018 hardware are ICE, and their engaged minutes under 2018 openpilot's own longitudinal control sit at a
  coin flip (`comma2k19`, 2.14).** One 2017 RAV4 and one 2016 Civic, 381 windows (6.4 h) decoded from the raw 2018 rlogs
  with `CarParams` built from the live fingerprint. Powertrain: ICE for both (P 0.68 / 0.71; 88.1 % of the 42 drives), the
  155 human-driven windows 100.0 % right (P(BEV) 0.06), the 226 engaged ones 50.9 % (0.47), and the speed-only variant is
  not better there (64.3 % of the windows). Size: the Civic is a coin flip with the full model (P(SMALL) 0.485 against
  P(MID) 0.491 -> MID), the RAV4 SMALL (0.54); the human-driven windows of both cars are SMALL (P 0.68-0.72), the engaged
  ones MID (0.58-0.61); the speed-only variant has both SMALL (97.6 % of the drives). Probes, not fleet statements.
* **The steering-derived yaw of the three CAN sets is inflated at speed.** The 2018 logs carry a `carState.yawRate` (the
  output of 2018 openpilot's single-track vehicle model, not a sensor; 2.14, `UNDERSTEER.md`): the
  decoder's bicycle-model yaw matches it at 1-5 m/s (ratio 0.98 / 0.87, RAV4 / Civic) and overestimates it as 1 + k v^2
  with k of about 0.001 s^2/m^2 (1.13 / 1.03 at 10-15 m/s, 1.54 / 1.53 at 20-25, 1.85 / 1.78 at 25-30, 2.06 / 2.01 at
  30-40; correlation 0.999), i.e. understeer the kinematic model ignores. Yaw, curvature and lateral acceleration are
  therefore 1.5-2x too large on the highway windows of `ccs`, `ccs_honda` and `comma2k19` relative to the lab's IMU / pose
  yaw, which bounds the lateral signal's transfer from below (3.6 point 6) and is a candidate explanation of the yaw-mask
  sensitivity and of the full model's size loss on `ccs`; the remedy (an understeer-corrected or measured-yaw re-decode of
  the three sets) was the suggested next experiment of section 4 and is done in the block below (2.15).
* **What the two sets move.** Only the ICE and SMALL totals: the phase-3 scope goes from 2,232 / 2,547 to 2,460 / 2,778
  (88.6 %) for powertrain and from 2,245 / 4,152 to 2,473 / 4,384 (56.4 %) for size; comma2k19 adds 2 of 2 ICE and 1 of 2
  SMALL to "all sets" only. Two artefacts shared by the three sets are stated once in 2.13: sample 0 of every window is
  the fresh parser's all-zero `CarState`, and the lead gate admits occasional oncoming traffic.

Understeer-corrected re-decode of the three steering-derived sets (2.15, 3.7; post hoc, approved after the write-up above; the
pooled totals above are the as-run decodes and stay the primary numbers of this report):

* **The derived-yaw bias is fitted on the lab's own data and removed with one global constant.** On the lab's released drive
  tables the IMU / pose yaw against the steering angle (6,657 usable drives, 78 eligible platforms) shows the bicycle-model yaw
  overestimating the measured one as 1 + k v^2; the hours-weighted median over the 62 train / validation platforms (423.8 h),
  k = 0.00079289 s^2/m^2 (characteristic speed 35.5 m/s), is the ONE value used for every platform, so no platform-specific lab
  calibration enters the external features and the external labels were never used (`UNDERSTEER.md`). The three sets were
  re-decoded with `ccs_decode.py --yaw derived-understeer --understeer-k 0.00079289` and nothing else changed: same rows, same
  labels, every non-yaw channel identical, the yaw-masked runs of both decodes identical. The comma2k19 ratio that motivated
  the correction compares the bicycle model with 2018 openpilot's own vehicle-model `yawRate`, not with a sensor (2.14).
* **Size on the device-disjoint fleet moves from 54.1 to 63.2 % of 4,061 devices (+9.1; macro-F1 46.4 -> 60.3), against the
  internal held-out vehicle accuracy of 63.5 / 59.7.** Models 63 -> 72 of 113, platforms 95 -> 108 of 161 (27 changed decision:
  15 LARGE platforms recovered and none lost, 4 MID recovered and 2 lost, 4 SMALL lost, 2 LARGE wrong -> wrong); devices right
  LARGE 241 -> 532 of 1,025 (23.5 -> 51.9 %; internal LARGE vehicle recall 50.6), MID 351 -> 442 of 1,274 (27.6 -> 34.7; internal
  47.3), SMALL 1,606 -> 1,594 of 1,762 (91.1 -> 90.5; internal 79.2); seen 53.9 -> 64.3 % of devices, held-out 53.8 -> 58.0 %,
  absent 88.9 -> 88.9. The minivans and three-row SUVs come back (all four Pacifica platforms, Palisade, CX-9, Sienna 4th gen,
  with the Rivian R1, Maverick, Ranger, G80, Hyundai Genesis, both Escalades and the Crafter); the Honda Odyssey / Pilot /
  Ridgeline, Carnival, Atlas, G90 / GV80, Explorer and the older Sienna stay missed, and MID stays the weak class (8 of 33 models
  before and after). On this evidence the "nine points lost" of the phase-3 block were a speed-dependent bias of the decoded
  input rather than a limit of the lateral signal; the per-class gaps that remain are stated in 2.15.
* **Powertrain is essentially unchanged (87.8 -> 88.9 % of devices, 86 -> 85 of 88 models: the F-150 Lightning goes from P(BEV)
  0.52 to 0.46 on 6 devices), and the two SMALL-only sets move inside their single class** (`ccs_honda_us` 98.3 -> 93.5 % of 230
  devices SMALL, 11 devices fewer in SMALL (10 to MID, 2 to LARGE, one back from MID), every platform still SMALL; powertrain
  99.1 -> 99.6; `comma2k19_us` 1 of 2 SMALL and 2 of 2
  ICE as before). The `*_us` sets are excluded from every pooled table by design (they re-decode the same windows), so no total of
  this section, 3.1 or 3.2 changes; `--yaw derived-understeer` is the recommended pipeline for future external decodes (section 4).

Phase 2 (two sets with steering channels and the car's own CAN acceleration; 2.10, 2.11, 3.5):

* **With steering present, size transfers to the level of our own held-out test (commaSteeringControl, 2.10).** 36
  platform archives of lateral-controller-engaged highway segments (51,462 windows, 857.7 h, 25 car models, one window
  per segment, CAN `aEgo`): the matched V1 variant (lateral + speed) reaches 63.9 % [50.0, 80.6] accuracy / 62.3 macro-F1
  over the 36 archives (one archive = one pseudo-vehicle; internal V1 vehicle reference 62.6 / 58.0; chance 33.3) and
  64.0 % over the 25 car models (16 of 25): SMALL 15 of 17 archives, MID 4 of 13, LARGE 4 of 6 (Silverado 1500, RAM
  1500, Pilot, Sienna; the Odyssey and the Palisade are called SMALL). On the 10 archives of held-out test models it is
  7 of 10 (Genesis GV60 MID and Toyota Sienna LARGE right; Kona Electric, both Leaf and both Corolla archives SMALL;
  misses: both Santa Fe archives SMALL, Ioniq Electric MID). Window level 48.4 % / 44.1 (internal 62.0 / 55.0). So the
  phase-1 size failure was about the inputs (no steering, derived accelerations), not about the size signature: LARGE
  is recognised for pickups and the Sienna on a fleet of unknown devices, MID stays the weak class (4 of 13 archives, as
  internally with 43 % vehicle-level MID recall), and the speed-only variant on the same windows gets no MID and 3 of 6
  LARGE.
* **The speed-only powertrain model transfers almost perfectly when the acceleration is the car's own CAN estimate.**
  On the same 36 archives the V3 variant calls 35 of 36 right (BEV 8 of 9, ICE 27 of 27; archive-level accuracy
  97.2 % [91.7, 100.0] / macro-F1 96.1, window 75.4 / 69.0; internal V3 vehicle reference 91.0 / 90.1), the held-out
  stratum 9 of 10 (both Leaf archives BEV, which the internal V3 misses on our own Leaf); the one miss is the Ioniq
  Electric (497 windows, P(BEV) 32 %). ICE cars under lateral automation on highways are ICE here, so the phase-1 ICE
  misses were the GNSS / derived-acceleration artefact of 3.3, not an automation effect. The matched V1 variant (lateral
  + speed) calls every archive ICE (BEV 0 of 9, window BEV recall 24.4 %), and so does the full model under the V1
  mask: the lateral channels hurt the powertrain decision on a new fleet, as VARIANTS.md predicted (internal V1 74.6
  drive accuracy).
* **With all 20 channels on a community-fork fleet (MoreTorque, 2.11) SMALL is right for all 8 SMALL cars and MID for
  none of the 4.** The full models on 7,190 windows (119.8 h, 12 cars, 1,155 drives): Mazda 3 SMALL for 7 of 7 cars
  (drive accuracy 91.7 %), Omoda E5 SMALL; the Hyundai Sonata (one car of a training platform, P(MID) 23 %) and the
  three Mazda CX-50 (P(MID) 22 %) are SMALL, so vehicle accuracy 66.7 % is the SMALL share. Masking the yaw group turns
  the Sonata and the CX-50 into LARGE calls: the lateral channels carry the decision but do not say MID on these forks,
  in no speed regime and in no automation state. Powertrain: the 11 ICE cars are ICE (drive accuracy 90.6-95.5 %), the
  one BEV (Omoda E5: one car, one driver, urban, pedal and torque flags absent) is ICE for every model (P(BEV) 20 % full
  model, 41 % V3), a single-car miss of the kind of the internal Tesla Model S.
* **Pooled over phases 1 and 2** (3.1, 3.2; one physical car or archive = one decision): BEV 31 / 37, ICE 306 / 322; size
  SMALL 48 / 52, MID 6 / 37, LARGE 4 / 19, where every MID and LARGE hit outside the two Tesla Model 3 comes from the
  steering-equipped commaSteeringControl set (the totals including phase 3 are in the phase-3 block above).

Phase 1 (nine sets: eight without steering channels and the single-car A2D2 probe; GNSS / OBD / derived accelerations;
2.1-2.9):


* **The BEV call transfers, the ICE call does not survive controller-driven, GNSS-sensed driving.** Over the eight
  sets with a BEV / ICE label, 23 of the 24 named BEV cars outside VED are called BEV at vehicle level by the primary
  run of their set (the matched reduced-input variant; the full model for the commaCarSegments sample): every Tesla
  Model 3 / S / X, Audi e-tron, Jaguar I-Pace, VW e-Golf, Ioniq 5, Polestar 2, both Kia Niro EV, Bolt EUV, Rivian R1;
  the miss is the Ford Mustang Mach-E. Only 10 of the 20 named ICE cars are called ICE: the human-driven platoon
  leaders (Polo, Space Star, Rexton, MINI), the Audi A8 leader on cruise control, the Ford S-Max (an ACC follower in
  83 % of its windows, ICE in both platoon sets) and the Fiat 500X of the one-vehicle campaign (99.1 % of 429 trips,
  20 drivers). The ICE cars that follow a lead under adaptive cruise on a test track (Audi A6, BMW X5, Mazda 3, Volvo
  XC40 in both sets; the Mercedes A-Class in OpenACC, a coin flip at 52-53 % in both sets; the Toyota RAV4 of the
  commaCarSegments sample) and the scripted UGA Lexus UX 200 lead car are called BEV.
* **Best external powertrain numbers, matched variant (V2, longitudinal-only):** OpenACC platoons (20 named models,
  19 cars, 75.7 h) vehicle accuracy 73.7 % [52.6, 94.7] / macro-F1 73.7 (BEV recall 100.0, ICE 58.3), drive level 80.3 %
  [64.2, 90.6] / 76.3 against our own 91.4 / 90.4 (vehicle) and 90.0 / 89.0 (drive) for the same variant; ULTra-AV
  (11 named models) vehicle 75.0 % / 69.7 (BEV 100.0, ICE 40.0). Yuan et al. Tesla Model 3 (speed only, V3): 80.2 %
  of 106 urban trips called BEV (our V3 BEV drive recall 77.2 on human-only windows). VED: 258 of 264 ICE cars called
  ICE (97.7 %), but the three 2013 Nissan Leafs are called ICE (16.4 % of their 317 trips BEV), exactly as the speed-only
  variant treats the held-out Leaf on our own data (18 % of drives, model-level ICE).
* **Size does not transfer beyond SMALL on these sets.** Pooled over the phase-1 sets, 25 of 27 named SMALL cars, 2 of 20 MID cars and 0 of
  13 LARGE cars are called right at vehicle level (the two MID hits are Tesla Model 3s, a training platform). On the
  longitudinal / speed-only sets the speed-only variant V3 says SMALL for almost everything (ULTra-AV vehicle accuracy
  36.4 %, macro-F1 17.8; OpenACC 38.1 % / 18.4; chance 33.3; our own V3 reference 54.4 / 45.6), every LARGE car
  (Model S, Model X, I-Pace, X5, A8, GLE 450) included; the A2D2 Audi Q7 e-tron probe with the full 20-channel model is
  called SMALL with 85 % confidence (0 of 34 windows); the Rivian R1 sample window is called MID. The VED weight-bin
  ordinal check shows no relation (Spearman -0.125 over 368 cars).
* **Why (diagnostic on our own test windows):** every external set but commaCarSegments has to derive the
  acceleration from the speed. Replacing the CAN `aEgo` of our own held-out windows by that derivative drops the
  speed-only variant's LARGE drive recall from 28.7 to 3.4 % (GNSS-like smoothing: 0.8 %) and its BEV drive recall
  from 76.5 to 45.1 %; the longitudinal variant keeps its powertrain accuracy under a 0.5-s derivative (86.3 vs 87.4)
  but loses ICE recall under smoother speeds (76.6 % GNSS-like, 46.0 % at 1.0 s, 27.5 % VED-like). The speed-only size
  signal and part of the powertrain signal live in the 10 Hz texture of the vehicle's own acceleration estimate, which no
  external dataset provides.
* **Secondary numbers (full 20-channel model with the missing channels at the training missing value) are not
  informative on reduced inputs:** on our own test windows the full model under the V2 / V3 masks calls almost every
  drive SMALL (MID recall 1-5 %, LARGE 0-2 %) and 82-83 % of the BEV drives ICE, and it does the same externally (Tesla Yuan: 0 of 106 trips BEV
  under the V3 mask against 80.2 % with the V3 variant). They are listed for completeness only.

## 1. Set-up

### 1.1 Window sets and the model used for each

| set (`<external>/<set>/work`) | what it is | channels present (of 20) | primary model | secondary | reference subset |
|---|---|---|---|---|---|
| `ultraav` | ULTra-AV named-vehicle step3 subset: 2,976 windows (49.6 h), 20 models, 22 cars, 121 drives; 97.5 % ACC-driven car following (ZalaZone / AstaZero / Vicolungo / Casale / CATS / Ohio) | 11: v, a (derived), standstill, das, lead group | V2 variant (both tasks); V3 variant for size (VARIANTS.md: V2 is at chance for size) | full model + V2 / V3 mask | das |
| `openacc` | OpenACC raw platoon campaigns: 4,540 windows (75.7 h), 20 models, 21 cars, 630 run series; 78 % ACC | 11, as above | V2 (powertrain), V3 (size) | full + V2 / V3 mask | das |
| `microsim_ev` | microSIM-ACC / EV: 111 windows (1.9 h), 3 BEV ACC followers (Ioniq 5, Polestar 2, Model 3) | 11 | V2 (powertrain), V3 (size) | full + V2 / V3 mask | das |
| `uga_ev` | UGA EV platoon: 326 windows (5.4 h), 4 cars (Lexus UX 200 lead, Mach-E, two Niro EV), 48 runs, followers on ACC in 8 of 12 runs | 11 | V2 (powertrain), V3 (size) | full + V2 / V3 mask | das |
| `tesla_yuan` | Yuan et al. 2024: one 2019 Tesla Model 3 SR+, 106 urban trips, 2,619 windows (43.7 h), human driving assumed | 4: v, a (derived), standstill, das = 0 | V3 variant | full + V3 mask | human |
| `openacc_500x` | OpenACC one-vehicle campaign: one Fiat 500X diesel, 20 drivers, 429 trips, 13,523 windows (225.4 h), manual | 5 (lead_valid = 0) | V3 | full + V3 mask | human |
| `ved` | VED: 384 cars (264 ICE, 93 HEV, 24 PHEV, 3 BEV), 59,563 windows (992.7 h) from 1 Hz OBD speed upsampled to 10 Hz, capped at 200 windows per car | 4 | V3 | full + V3 mask | human |
| `a2d2` | A2D2 bus signals: one Audi Q7 e-tron (diesel PHEV, LARGE 5.045 m^2), 34 windows (0.6 h), manual urban driving | 11: v, a, steering angle / rate, yaw, curvature, lateral acceleration, gas, brake, standstill, das = 0; lead and torque absent (training dropout encoding); steer_pressed unknown | full 20-ch model | V1 variant, full + V1 mask, V3 variant, full + V3 mask | human |
| `ccs_sample` | commaCarSegments, the 5 verification segments -> 4 windows (Model 3, Rivian R1, RAV4 TSS2, Bolt EUV) | 20 (lead only on the RAV4; torque on two cars) | full 20-ch model | - | all |
| `csc` (phase 2) | commaSteeringControl: 36 platform archives of 60-s segments with the openpilot lateral controller active (comma, 2023), 1,500 segments per archive -> 51,462 windows (857.7 h), 25 car models, 9 BEV + 27 ICE archives; no device ids (one archive = one pseudo-vehicle, one window per segment); CAN `aEgo`, steering angle, localizer lateral acceleration; 92 % of the samples engaged, mean speed 24.6 m/s | 11: v, a (CAN), steering angle / rate (difference), yaw / curvature / lateral acceleration (localizer), steer_pressed, standstill, das = lateral controller active, lead_valid = 0 (= the V1 list) | V1 variant (size); V3 variant (powertrain, see 1.3) | full + V1 mask; V3 (size); V1 (powertrain) | das |
| `moretorque` (phase 2) | MoreTorque/rlogs: 7,636 one-minute openpilot rlogs of four community-fork platforms -> 7,190 windows (119.8 h), 1,155 drives (runs of consecutive segments), 12 cars: Hyundai Sonata 2020 (1 car, training platform), Mazda 3 2019 (7, absent), Mazda CX-50 (3, absent), Omoda E5 (1 BEV, absent); the lab drive builder reproduced from the rlogs | 20 (vision lead, localizer yaw, driver torque, pedal flags; 17 on the Omoda: torque and pedals absent) | full 20-ch model | V1 / V2 / V3 variants, full + V1 / V2 / V3 masks, lead / yaw / torque+pedal masks, `--das any` re-cut | all |
| `ccs` (phase 3) | commaCarSegments, device-disjoint selection (`CCS_SELECTION.md`): 49,563 one-minute `rlog.zst` segments of 4,334 devices that are absent from the lab's raw library -> 47,006 decoded (`ccs_decode.py`, opendbc 0.3.1) -> 45,763 windows (762.7 h), 21,987 drives (device x route), 4,061 devices, 113 car models, 161 platforms; one window per segment; strata by car model: seen (70 models, 3,199 devices), held-out (29 of the lab's 41 test models, 864 devices), absent (14 models, 36 devices); 39.3 % of the windows automation-engaged, mean speed 18.0 m/s | 20 where the platform provides them (torque absent on 15.4 % of the windows; lead from radar tracks on 34.0 %, the stock-ACC object on 10.4 %, absent on vision-only platforms 55.7 %; yaw / curvature / lateral acceleration steering-derived everywhere) | full 20-ch models (both tasks) | V1 / V2 / V3 variants, full + V1 / V2 / V3 masks, yaw and lead masks (sensitivity) | all; the engaged / human split inside the set |
| `ccs_honda` (phase 3, addendum) | commaCarSegments, the nine Honda platforms the phase-3 decode lost (`CCS_PIPELINE.md` "Honda re-decode"): the 2,198 budget-list segments of HONDA_CIVIC / CIVIC_BOSCH / CIVIC_2022 / CRV / CRV_5G / FIT / HRV / HRV_3G / FREED re-decoded with `ccs_decode.py --honda-gearbox-from-can` -> 2,189 decoded -> 2,132 windows (35.5 h), 913 drives, 230 devices, 5 car models; strata seen (Civic, Fit, CR-V: 224 devices), held-out (HR-V, 5), absent (Freed, 1); every platform SMALL, every labelled one ICE (the Freed unlabelled); 9 CIVIC_2022 devices with an `automatic` gearbox message (possible hybrids) kept under the ICE label; 42.3 % of the windows engaged, mean speed 18.2 m/s | 20 where the platform provides them (Nidec radar lead on five platforms, 41.6 % of the windows; none on the four radarless Bosch ports; torque absent on 19.0 %, on every HONDA_CRV window; yaw steering-derived) | full 20-ch models (both tasks) | V1 / V2 / V3 variants, full + V1 / V2 / V3 masks, yaw and lead masks | all; the engaged / human split inside the set |
| `comma2k19` (phase 3, addendum; own scope) | comma2k19 (comma.ai 2018, MIT): Chunk 1 = one 2017 Toyota RAV4 TSS-P (TOYOTA_RAV4, seen SMALL ICE), Chunk 3 = one 2016 Honda Civic Touring (HONDA_CIVIC, seen SMALL ICE); 390 raw 2018 rlog segments in 42 route folders -> 387 decoded by `ccs_decode.py` with `--layout comma2k19 --fingerprint-map --das-from controlsState --ignore-absent-messages` -> 381 windows (6.4 h; 185 / 196), 42 drives, 2 devices; `CarParams` built from the live fingerprint (no logged one in any segment); 59.3 % of the windows engaged, with 2018 openpilot doing the longitudinal control itself; mean speed 22.8 / 20.9 m/s | 20 on every window (TSS-P / Nidec radar lead, `lead_valid` 0.78 / 0.87; torque scale never absent; yaw steering-derived; a `yawRate` logged by 2018 openpilot's vehicle model, not a sensor, exists in the logs and was used for the check of 2.14) | full 20-ch models (both tasks) | V1 / V2 / V3 variants, full + V1 / V2 / V3 masks, yaw and lead masks | all; the engaged / human split inside the set |
| `ccs_us` (understeer re-decode, 2.15) | `ccs` re-decoded by `ccs_decode.py --yaw derived-understeer --understeer-k 0.00079289` (`CCS_PIPELINE.md` "Understeer-corrected yaw", `UNDERSTEER.md`), every other option as the original decode: same budget list, 49,563 found -> 47,006 decoded / 2,557 skipped -> 45,763 windows (762.7 h) of the same 4,061 devices, 21,987 drives, 113 models and 161 platforms with the same labels and strata; 2.9 h with 6 workers; the collector verifies the alignment (same rows, every non-yaw channel identical) | 20 as `ccs`; yaw / curvature / lateral acceleration = the bicycle yaw divided by 1 + k v^2 | full 20-ch models (both tasks), read original -> corrected against `ccs` | the same nine runs as `ccs` | excluded from every pooled table; 2.15 only |
| `ccs_honda_us` (understeer re-decode, 2.15) | `ccs_honda` re-decoded with the same yaw option and `--honda-gearbox-from-can` as the original: 2,189 decoded -> 2,132 windows (35.5 h), 913 drives, 230 devices, 9 platforms, 5 models, same labels and strata; 10 min | 20 as `ccs_honda`; yaw group corrected | full 20-ch models, original -> corrected against `ccs_honda` | the same nine runs | excluded from every pooled table; 2.15 only |
| `comma2k19_us` (understeer re-decode, 2.15) | `comma2k19` re-decoded with the same yaw option and the three comma2k19 options of the original: 387 decoded -> 381 windows (6.4 h), 42 drives, 2 devices; 3 min | 20 as `comma2k19`; yaw group corrected | full 20-ch models, original -> corrected against `comma2k19` | the same nine runs | excluded from every pooled table; 2.15 only |

ULTra-AV and the raw OpenACC set are not independent: ULTra-AV's named-vehicle episodes are a harmonised subset of the
same JRC campaigns (plus CATS / Ohio), so they are two views of largely the same recordings. Every European model in
them (Audi, BMW, Mercedes, Jaguar, Volvo, Fiat, MINI, SsangYong, Mitsubishi, VW Polo / e-Golf) is absent from the lab
inventory; Tesla Model 3, Ioniq 5, Niro EV, Mach-E, Bolt EUV, Rivian R1 and RAV4 are training platforms, Tesla Model S
and Nissan Leaf are held-out test models of `split_platform`.

Phase-2 strata (by car model against `build/index/drives.parquet`): in commaSteeringControl 26 archives (18 models) are
training / validation models (`seen`) and 10 archives (7 models: Genesis GV60, Hyundai Ioniq Electric, Kona Electric,
Santa Fe, Nissan Leaf, Toyota Corolla, Sienna) are held-out test models of `split_platform` (`held-out`); none is absent
from the inventory. In MoreTorque the Sonata is a training platform (24 lab drives), the Mazda 3 (BP), CX-50 and Omoda E5
are absent (the Mazda brand is seen through the CX-5, CX-9 and 6).

Phase-3 strata (by car model, same index): in commaCarSegments every device is absent from the lab's raw library by
construction, so `seen` (70 models / 100 platforms / 25,667 windows) is "training platform, new device", `held-out` (29 of
the 41 `split_platform` test models / 47 platforms / 18,710 windows) the external replica of the lab's held-out test, and
`absent` (14 models / 14 platforms / 1,386 windows, footprints from public specs) the out-of-inventory probe; the Honda
Civic, Fit, HR-V and Freed of the selection are missing because their segments did not decode (2.12). They were recovered
on 10 Nov as the second phase-3 set `ccs_honda` (2.13: Civic, Fit and the gasoline CR-V ports as `seen`, 224 devices; the
HR-V `held-out`, 5; the Freed `absent`, 1), scored with the same nine runs and pooled with `ccs` in 3.1 / 3.2; `comma2k19`
(2.14, two seen SMALL ICE cars on 2018 hardware) is a scope of its own.

### 1.2 In-distribution references (our held-out car models, seed ensembles; `internal_reference.md`)

Accuracy / macro-F1 in percent, recall per class in brackets.

| task | model | window (all) | drive (all) | vehicle (all) | drive (das) | drive (human) | vehicle (das) | vehicle (human) |
|---|---|---|---|---|---|---|---|---|
| BEV vs ICE | full ResNet B5 (20 ch) | 82.5 / 80.7 (BEV 76.2, ICE 85.7) | 93.7 / 93.3 (88.7, 96.9) | 93.9 / 93.4 (89.2, 96.7) | 92.3 / 91.9 (87.5, 95.6) | 90.2 / 89.3 (88.1, 91.4) | 93.2 / 92.8 (89.2, 95.8) | 90.8 / 90.2 (87.3, 92.9) |
| BEV vs ICE | V1 lateral + speed (10 ch) | 70.0 / 67.9 (66.2, 71.9) | 74.6 / 73.7 (70.6, 77.2) | 75.8 / 74.6 (71.0, 78.8) | 73.9 / 73.6 (77.9, 71.1) | 73.8 / 71.5 (66.8, 77.4) | 77.6 / 76.8 | 72.3 / 70.7 |
| BEV vs ICE | V2 longitudinal-only (11 ch) | 76.7 / 74.1 (66.2, 82.0) | 90.0 / 89.0 (78.3, 97.4) | 91.4 / 90.4 (77.4, 100.0) | 86.9 / 86.3 (78.3, 93.0) | 86.0 / 84.2 (76.7, 90.8) | 85.4 / 84.2 (74.3, 92.4) | 89.8 / 88.7 (75.9, 98.4) |
| BEV vs ICE | V3 speed-only (3 ch) | 77.0 / 74.6 (68.2, 81.5) | 89.5 / 88.6 (78.9, 96.3) | 91.0 / 90.1 (79.6, 98.0) | 88.3 / 87.9 (84.6, 90.9) | 87.6 / 85.9 (77.2, 93.0) | 86.5 / 85.6 (81.1, 89.8) | 89.8 / 88.6 (74.7, 99.2) |
| BEV vs ICE | full + V2 mask | 71.0 / 60.9 (29.8, 92.0) | 67.5 / 54.4 (17.7, 99.4) | 69.3 / 56.8 (20.4, 99.3) | 66.0 / 53.7 | 72.2 / 59.1 | 66.7 / 51.2 | 69.9 / 58.5 |
| BEV vs ICE | full + V3 mask | 69.7 / 59.4 (28.6, 90.8) | 63.3 / 51.0 (16.8, 93.1) | 65.2 / 49.8 (12.9, 97.4) | 69.4 / 60.7 | 64.0 / 49.4 | 70.3 / 59.0 | 63.6 / 50.1 |
| size | full GRU S9 (20 ch) | 60.0 / 54.3 (SMALL 71.7, MID 48.2, LARGE 42.5) | 70.8 / 67.3 (80.6, 61.3, 58.6) | 63.5 / 59.7 (79.2, 47.3, 50.6) | 70.2 / 62.5 (85.6, 59.7, 38.6) | 63.5 / 62.6 (66.9, 61.0, 60.3) | 63.4 / 55.6 | 59.5 / 58.6 |
| size | V1 lateral + speed | 62.0 / 55.0 (78.7, 38.3, 45.5) | 70.6 / 66.4 (83.9, 50.0, 62.5) | 62.6 / 58.0 (80.5, 43.0, 49.4) | 65.1 / 57.4 | 67.6 / 65.9 (74.9, 56.4, 66.3) | 62.6 / 54.9 | 64.3 / 62.4 |
| size | V2 longitudinal-only | 41.7 / 34.9 (55.8, 18.4, 31.6) | 45.2 / 33.5 (72.1, 9.2, 23.1) | 43.2 / 32.8 (73.6, 10.8, 19.5) | 46.8 / 34.3 | 41.3 / 36.4 | 46.2 / 36.0 | 39.0 / 34.5 |
| size | V3 speed-only | 47.1 / 41.3 (58.5, 36.7, 29.1) | 53.4 / 45.1 (73.4, 37.7, 23.9) | 54.4 / 45.6 (82.4, 32.3, 23.4) | 49.4 / 39.3 (69.1, 25.7, 22.2) | 50.4 / 46.5 (66.3, 43.6, 30.2) | 56.1 / 47.0 | 48.3 / 44.1 |
| size | full + V2 mask | 53.6 / 33.2 (89.4, 10.2, 9.3) | 52.7 / 24.8 (99.2, 1.4, 1.6) | 48.0 / 23.2 (98.1, 0.0, 2.6) | 56.6 / 28.3 | 46.0 / 22.2 | 51.1 / 26.3 | 46.5 / 23.0 |
| size | full + V3 mask | 55.1 / 30.1 (95.3, 8.7, 2.9) | 52.8 / 26.1 (98.2, 4.8, 0.4) | 49.2 / 25.3 (98.7, 4.3, 1.3) | 56.0 / 25.5 | 47.1 / 26.1 | 51.5 / 24.6 | 46.8 / 24.7 |

Per held-out model (`internal_per_model.md`): the full size model gets 14 of 15 SMALL, 6 of 14 MID and 8 of 12 LARGE
models right at model level, V3 14 / 3 / 3 and V2 14 / 0 / 3; the LARGE hits of V3 are the Ford Ranger, GMC Yukon and
Genesis GV80, its misses include Tesla Model S, Genesis G80 / G90, VW Atlas, Toyota Sienna, Kia Carnival / EV9 and
the F-150 Lightning. For BEV vs ICE the full model, V2 and V3 get 8 of 9 BEV and 30 of 30 ICE models (V1: 7 of 9 and
26 of 30); the BEV miss of V1 / V2 / V3 is the Nissan Leaf (16-21 % of its drives; V1 also misses the Tesla Model S and
four ICE models), the miss of the full model is the Tesla Model S.

### 1.3 Reading rules

* The primary number of a set is the variant whose input list equals the channels the set provides (V2: speed,
  acceleration, standstill, automation flag, lead group; V3: speed, acceleration, standstill); for size on the
  longitudinal sets V3 is the reference because V2 size is at chance on our own data (VARIANTS.md). The full model
  with the same channels masked is the secondary number; section 3.4 shows why it cannot be read as a result.
* Drive-level numbers on ULTra-AV / OpenACC are dominated by the many short ZalaZone BEV runs (ULTra-AV: 75 of 84
  labelled drives are BEV); the vehicle and model levels are the honest ones, and every model is one physical car
  except the Tesla Model 3 (two) and the Kia Niro EV (two).
* Borderline footprints (|footprint - cut| < 0.05 m^2): Audi A4 Avant (4.41), Audi e-tron (4.84), Jaguar I-Pace
  (4.92), Audi A8 (4.91), Ioniq 5 (4.83), Polestar 2 (4.38), Mustang Mach-E (4.85). The size metrics are repeated
  without them where it changes anything. Phase 2 adds the Camry 2018 / 2021 (4.47), the Pilot (4.92) and, by the
  `specs_external` flag, the Ioniq 5 (4.83) in commaSteeringControl; MoreTorque has no borderline model.
* Phase 2: on commaSteeringControl the powertrain primary is the speed-only variant V3, not the matched V1, by the same
  reasoning that made V3 the size reference on the longitudinal sets: V1 is the one variant that is weaker than the
  speed-only inputs for the powertrain task on our own data (drive accuracy 74.6 against 89.5, VARIANTS.md: the lateral
  channels let the recipe fit car models instead of the drivetrain), so V1 would measure that weakness rather than the
  transfer. V1 and the full model under the V1 mask are reported next to it. For size the matched V1 is the primary
  (it equals the full model internally). On commaSteeringControl one window is one segment and no device ids exist,
  so "drive" = window and "vehicle" = archive (= 2023 platform); the bootstrap resamples archives.
* Phase 3: on commaCarSegments all 20 channels are present (with the lead and torque groups absent where the platform
  has none, the training dropout encoding), so the full 20-channel models are the primary run for both tasks and the
  variants and masks are comparisons; `vehicle` = device pseudonym, `drive` = device x route, one window per segment;
  the per-stratum intervals are vehicle-cluster bootstraps of 300 resamples (the score files use 1,000 on the pooled
  set); the `mask-yaw` and `mask-lead` runs are sensitivity checks for the two channel groups whose construction differs
  from training (steering-derived yaw; radar-gate / stock-ACC lead), with no internal reference. A device that appears
  under more than one platform (91 of 4,061, three of them under three) carries its majority label in the level tables and
  counts once per model in the per-model and pooled-decision tables; the per-model tables judge a device against its
  model's majority label, which differs from the device's own platform label for 13 device x model pairs (Acura RDX and
  Lexus IS, whose platforms fall in two size classes; Kona, EV and gasoline ports), while the pooled-decision tables of
  3.1 / 3.2 and of `RESULTS_TABLES.md` use the device's own label (3.1).
* Phase 3, addenda: `ccs_honda` is read exactly like `ccs` (full models primary, `vehicle` = device pseudonym, one window
  per segment, 300-resample strata intervals); its devices carry the same `ccs_v_` pseudonyms as `ccs`, 9 of them appear in
  both sets under different platforms and count once per set x car model in the pooled-decision tables; its 9
  `automatic`-gearbox HONDA_CIVIC_2022 devices (most likely the Civic Hybrid filed under the same opendbc name) keep the
  platform's ICE label, and the powertrain numbers are given with and without them. Both tasks have a single class there
  and in `comma2k19`, so the accuracies are the SMALL / ICE recall of each run, no macro-F1 exists, the majority baseline
  is 100 %, and the masked full models (which call almost everything SMALL / ICE, 3.4) score as high as the full model
  without that meaning anything. `comma2k19` (two cars, one driver each, 2018 software) is a scope of its own in the
  pooled tables, outside the phase-3 device scope and inside "all sets"; its decisions are single-car probes.
* Understeer-corrected re-decodes (2.15): `ccs_us`, `ccs_honda_us` and `comma2k19_us` are the same windows as `ccs`, `ccs_honda`
  and `comma2k19` with the three yaw-group channels re-derived (the bicycle yaw divided by 1 + k v^2 with one global k fitted on
  the lab's own IMU / pose yaw, never on the external labels) and are excluded from every pooled table, the summary table and
  the per-dataset blocks of `RESULTS_TABLES.md` by design; they appear only in its last section and in 2.15. The as-run numbers
  of 2.12-2.14 remain the primary results of this report (that decode was planned and run before the bias was known); the
  corrected numbers are quoted as "corrected" next to them, original -> corrected at device and model level with the same
  aggregation rules, and `--yaw derived-understeer` is the recommended pipeline for future external decodes rather than a
  replacement of the as-run tables. The yaw-masked runs of both decodes must be identical; that equality is the check that
  nothing but the yaw group changed.

## 2. Results per dataset

### 2.1 ULTra-AV (`ultraav`)

What it is: the harmonised car-following CSVs of ULTra-AV (CATS Lab) restricted to the 12 files with named follower
vehicles at 10 Hz; 20 models / 22 cars / 121 drives, 2,976 windows; one physical car per model; 97.5 % of the samples
under adaptive cruise (73 human-driven windows); a lead vehicle on 99.99 % of the samples; speeds are GNSS (RT-Range at
AstaZero, splines of 3-5 Hz u-blox elsewhere); acceleration is the 0.5-s smoothed derivative of the speed (`DS_ULTRAAV.md`).
Labels: 1,330 BEV windows (6 models, 7 cars), 657 ICE (5 models), 989 HEV / PHEV / FCEV / unknown (size only); size
919 SMALL (8 models) / 944 MID (7 models, 9 cars) / 1,113 LARGE (5 models).

Powertrain (BEV vs ICE):

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall BEV / ICE % | chance uniform / majority % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (primary) | window | 1,987 | 72.1 [54.8, 86.6] | 63.3 [51.0, 72.9] | 90.5 / 34.9 | 50.0 / 66.9 | 76.7 / 74.1; 75.3 / 73.0 |
|  | drive | 84 (75 BEV / 9 ICE) | 85.7 [46.1, 96.7] | 46.2 [31.5, 49.1] | 96.0 / 0.0 | 50.0 / 89.3 | 90.0 / 89.0; 86.9 / 86.3 |
|  | vehicle | 12 (7 / 5) | 75.0 [50.0, 100.0] | 69.7 [36.8, 100.0] | 100.0 / 40.0 | 50.0 / 58.3 | 91.4 / 90.4; 85.4 / 84.2 |
|  | model | 11 (6 / 5) | 72.7 [57.1, 100.0] | 68.6 [38.5, 100.0] | 100.0 / 40.0 | 50.0 / 54.5 | - |
| V3 variant | window | 1,987 | 52.0 [33.8, 71.2] | 51.4 [32.6, 65.2] | 30.6 / 95.3 | 50.0 / 66.9 | 77.0 / 74.6; 75.1 / 73.4 |
|  | vehicle | 12 | 41.7 [16.7, 66.7] | 29.4 [14.3, 40.0] | 0.0 / 100.0 | 50.0 / 58.3 | 91.0 / 90.1; 86.5 / 85.6 |
| full model + V2 mask (secondary) | window | 1,987 | 45.4 [33.6, 59.3] | 45.4 [31.9, 56.5] | 32.3 / 72.0 | 50.0 / 66.9 | 71.0 / 60.9; 71.4 / 60.9 |
|  | vehicle | 12 | 50.0 [25.0, 75.0] | 43.8 [20.0, 73.3] | 14.3 / 100.0 | 50.0 / 58.3 | 69.3 / 56.8; 66.7 / 51.2 |
| full model + V3 mask (secondary) | window | 1,987 | 35.1 [18.0, 56.2] | 32.4 [18.0, 44.8] | 11.1 / 83.7 | 50.0 / 66.9 | 69.7 / 59.4; 70.3 / 62.0 |
|  | vehicle | 12 | 41.7 [16.7, 66.7] | 29.4 [14.3, 40.0] | 0.0 / 100.0 | 50.0 / 58.3 | 65.2 / 49.8; 70.3 / 59.0 |

Named cars, powertrain, V2 variant (mean probability over the car's windows -> decision; drive accuracy in brackets):
BEV: Audi e-tron BEV 87.5 % (100.0), Jaguar I-Pace 84.9 % (97.8), Tesla Model 3 (two cars) 77.3 % (94.3), Tesla Model
S 78.9 % (91.9), Tesla Model X 74.0 % (94.0), VW e-Golf 52.6 % (83.3): 7 of 7 cars right. ICE: Ford S-Max P(ICE) 52.2 %
(42.9) and Mercedes A-Class 52.2 % (50.0) right; Audi A6 P(BEV) 70.4 % (0.0), BMW X5 60.4 % (12.5) and Mazda 3 69.5 %
(10.0) called BEV: 2 of 5. Cars outside the task (predicted BEV share): Audi A4 Avant (48-V mild hybrid) 81.8 %, BMW
i3s REx 78.8 %, Hyundai Nexo (fuel cell) 86.3 %, Toyota RAV4 Hybrid 82.9 %, Kia Niro PHEV 53.1 %, Mitsubishi Outlander
PHEV 37.7 %, Mercedes GLE 450 (mild hybrid) 41.9 %, Lincoln MKZ 30.2 %.

Size:

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall SMALL / MID / LARGE % | chance uniform / majority % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V3 variant (primary) | window | 2,976 | 29.7 [14.0, 49.4] | 19.1 [10.9, 26.1] | 88.8 / 5.5 / 1.5 | 33.3 / 37.4 | 47.1 / 41.3; 47.3 / 40.1 |
|  | drive | 121 | 21.5 [8.5, 33.3] | 14.0 [6.1, 18.8] | 100.0 / 4.1 / 0.0 | 33.3 / 40.5 | 53.4 / 45.1; 49.4 / 39.3 |
|  | vehicle | 22 (8 / 9 / 5) | 36.4 [18.2, 54.5] | 17.8 [10.3, 24.8] | 100.0 / 0.0 / 0.0 | 33.3 / 40.9 | 54.4 / 45.6; 56.1 / 47.0 |
|  | vehicle, no borderline | 19 (7 / 8 / 4) | 36.8 [15.8, 57.9] | 17.9 [9.1, 24.4] | 100.0 / 0.0 / 0.0 | 33.3 / 42.1 | - |
| V2 variant | window | 2,976 | 29.7 [19.8, 41.3] | 27.9 [19.5, 35.5] | 34.2 / 50.8 / 8.2 | 33.3 / 37.4 | 41.7 / 34.9; 43.0 / 35.0 |
|  | drive | 121 | 45.5 [30.5, 68.7] | 37.1 [26.7, 41.2] | 58.3 / 83.7 / 0.0 | 33.3 / 40.5 | 45.2 / 33.5; 46.8 / 34.3 |
|  | vehicle | 22 | 45.5 [27.3, 68.2] | 35.7 [21.1, 48.9] | 50.0 / 66.7 / 0.0 | 33.3 / 40.9 | 43.2 / 32.8; 46.2 / 36.0 |
| full model + V2 mask (secondary) | vehicle | 22 | 22.7 [4.5, 40.9] | 17.5 [4.4, 30.8] | 50.0 / 0.0 / 20.0 | 33.3 / 40.9 | 48.0 / 23.2; 51.1 / 26.3 |
| full model + V3 mask (secondary) | vehicle | 22 | 36.4 [18.2, 54.5] | 17.8 [10.3, 24.8] | 100.0 / 0.0 / 0.0 | 33.3 / 40.9 | 49.2 / 25.3; 51.5 / 24.6 |

Named cars, size, V3 variant (decision, P of SMALL / MID / LARGE in percent):

| car (footprint m^2) | truth | decision | P(S / M / L) % | drive acc % |
|---|---|---|---|---|
| Audi A4 Avant (4.41, borderline) | SMALL | SMALL | 57.9 / 34.8 / 7.4 | 100.0 |
| BMW i3s REx (4.07) | SMALL | SMALL | 53.3 / 34.0 / 12.7 | 93.5 |
| Kia Niro PHEV (4.29) | SMALL | SMALL | 70.0 / 13.4 / 16.6 | 100.0 |
| Mazda 3 (4.20) | SMALL | SMALL | 51.8 / 34.1 / 14.1 | 80.0 |
| Mercedes A-Class (4.25) | SMALL | SMALL | 66.7 / 27.6 / 5.7 | 100.0 |
| Mitsubishi Outlander PHEV (4.11) | SMALL | SMALL | 64.3 / 13.7 / 22.0 | 100.0 |
| Toyota RAV4 Hybrid (4.34) | SMALL | SMALL | 48.7 / 33.2 / 18.1 | 77.8 |
| VW e-Golf (4.02) | SMALL | SMALL | 77.9 / 11.8 / 10.3 | 100.0 |
| Audi A6 (4.75) | MID | SMALL | 66.2 / 26.7 / 7.1 | 0.0 |
| Audi e-tron (4.84, borderline) | MID | SMALL | 52.0 / 34.9 / 13.2 | 6.0 |
| Ford Fusion (4.53; 1 window) | MID | SMALL | 91.1 / 6.0 / 2.9 | 0.0 |
| Ford S-Max (4.57) | MID | SMALL | 61.6 / 18.3 / 20.1 | 0.0 |
| Hyundai Nexo (4.52) | MID | SMALL | 70.1 / 24.0 / 5.9 | 0.0 |
| Lincoln MKZ (4.50, two cars) | MID | SMALL | 46.3 / 38.5 / 15.1 | 7.4 |
| Tesla Model 3 (4.57, two cars) | MID | SMALL | 60.0 / 31.3 / 8.7 | 3.8 |
| BMW X5 (4.99) | LARGE | SMALL | 67.6 / 26.7 / 5.7 | 0.0 |
| Jaguar I-Pace (4.92, borderline) | LARGE | SMALL | 59.4 / 32.8 / 7.8 | 0.0 |
| Mercedes GLE 450 (5.02) | LARGE | SMALL | 60.1 / 31.9 / 8.0 | 1.5 |
| Tesla Model S (4.97) | LARGE | SMALL | 58.4 / 33.9 / 7.7 | 0.0 |
| Tesla Model X (5.06) | LARGE | SMALL | 55.1 / 33.4 / 11.6 | 4.0 |

The V2 size variant spreads its decisions differently (MID for 7 of the 20 models, among them the Audi A6, Tesla Model
3, Lincoln MKZ and Nexo, but also SMALL cars such as the A4 Avant, i3s and Niro PHEV; no LARGE call) and is at chance
on our own data, so its 45.5 % vehicle accuracy here is not evidence of a size signal.

Caveats: controller-driven longitudinal profiles (the ACC's, not a driver's) in 97.5 % of the samples; GNSS speeds
(splines of 3-5 Hz at Vicolungo / Casale / ZalaZone) and a derived acceleration that is smoother than the training
`aEgo`; one physical car per model and cars confounded with campaigns (ZalaZone 30-60 km/h test track, BEV-heavy;
AstaZero rural track; Vicolungo / Casale motorway); nine channels absent; the two Tesla Model 3 may be one car.

### 2.2 OpenACC platoon campaigns (`openacc`)

What it is: the six JRC platoon campaigns read from the raw files (AstaZero, ZalaZone, Cherasco, Vicolungo, Casale, JRC
low speed): 4,540 windows (75.7 h), 20 named models, 21 cars, 630 run series; 78 % of the samples under ACC with a
per-sample Human / ACC flag in five campaigns; the human-driven platoon leaders (Fiat 500X, VW Polo, Mitsubishi Space
Star, SsangYong Rexton, MINI Cooper at the motorway campaigns) add ICE cars without a lead; GNSS speeds, derived
acceleration, lead group from the inter-vehicle spacing with a 150-m radar-range emulation (`DS_OPENACC.md`). Labels:
1,880 BEV windows (6 models, 7 cars) vs 2,075 ICE (12 models), 585 HEV (the two mild hybrids, shares only); size
1,427 SMALL (9 models) / 1,342 MID (5 models, 6 cars) / 1,771 LARGE (6 models).

Powertrain:

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall BEV / ICE % | chance uniform / majority % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (primary) | window | 3,955 | 68.2 [54.8, 79.3] | 67.6 [53.3, 77.7] | 86.3 / 51.8 | 50.0 / 52.5 | 76.7 / 74.1; 75.3 / 73.0 |
|  | drive | 512 (330 / 182) | 80.3 [64.2, 90.6] | 76.3 [61.9, 85.6] | 93.9 / 55.5 | 50.0 / 64.5 | 90.0 / 89.0; 86.9 / 86.3 |
|  | vehicle | 19 (7 / 12) | 73.7 [52.6, 94.7] | 73.7 [50.4, 94.2] | 100.0 / 58.3 | 50.0 / 63.2 | 91.4 / 90.4; 85.4 / 84.2 |
|  | model | 18 (6 / 12) | 72.2 [55.6, 90.9] | 72.1 [54.5, 90.9] | 100.0 / 58.3 | 50.0 / 66.7 | - |
| V3 variant | window | 3,955 | 65.6 [51.4, 80.9] | 61.5 [50.4, 70.3] | 34.7 / 93.6 | 50.0 / 52.5 | 77.0 / 74.6; 75.1 / 73.4 |
|  | vehicle | 19 | 68.4 [47.4, 89.5] | 52.5 [32.1, 73.7] | 14.3 / 100.0 | 50.0 / 63.2 | 91.0 / 90.1; 86.5 / 85.6 |
| full model + V2 mask (secondary) | window | 3,955 | 50.5 [39.4, 63.4] | 48.5 [39.3, 55.8] | 32.1 / 67.3 | 50.0 / 52.5 | 71.0 / 60.9; 71.4 / 60.9 |
|  | vehicle | 19 | 36.8 [15.8, 57.9] | 26.9 [13.6, 36.7] | 0.0 / 58.3 | 50.0 / 63.2 | 69.3 / 56.8; 66.7 / 51.2 |
| full model + V3 mask (secondary) | window | 3,955 | 52.7 [36.3, 72.0] | 43.9 [34.3, 50.9] | 13.7 / 88.0 | 50.0 / 52.5 | 69.7 / 59.4; 70.3 / 62.0 |
|  | vehicle | 19 | 63.2 [42.1, 84.2] | 38.7 [29.6, 45.7] | 0.0 / 100.0 | 50.0 / 63.2 | 65.2 / 49.8; 70.3 / 59.0 |

Named cars, powertrain, V2 variant (P(BEV) %, decision, drive accuracy %; DAS share of the car's windows):

| car | truth | DAS % | P(BEV) % | decision | drive acc % |
|---|---|---|---|---|---|
| Audi e-tron (borderline footprint) | BEV | 90.3 | 79.0 | BEV | 88.1 |
| Jaguar I-Pace | BEV | 88.7 | 79.2 | BEV | 96.7 |
| Tesla Model 3 (two cars) | BEV | 84.4 | 78.7 | BEV | 96.6 |
| Tesla Model S | BEV | 89.8 | 78.8 | BEV | 98.6 |
| Tesla Model X | BEV | 90.4 | 72.6 | BEV | 94.2 |
| VW e-Golf | BEV | 90.1 | 50.7 | BEV | 57.1 |
| Audi A8 (leader on cruise, no lead) | ICE | 100.0 | 21.8 | ICE | 100.0 |
| Fiat 500X (human leader) | ICE | 0.0 | 30.0 | ICE | 83.3 |
| Ford S-Max | ICE | 83.1 | 43.0 | ICE | 72.2 |
| MINI Cooper (human) | ICE | 0.0 | 38.7 | ICE | 100.0 |
| Mitsubishi Space Star (human leader) | ICE | 0.0 | 28.5 | ICE | 100.0 |
| SsangYong Rexton (human leader) | ICE | 0.0 | 37.9 | ICE | 66.7 |
| VW Polo (human leader) | ICE | 0.0 | 30.1 | ICE | 100.0 |
| Audi A6 | ICE | 79.1 | 72.0 | BEV | 0.0 |
| BMW X5 | ICE | 79.1 | 63.8 | BEV | 10.0 |
| Mazda 3 | ICE | 88.9 | 54.4 | BEV | 42.9 |
| Mercedes A-Class | ICE | 79.1 | 53.2 | BEV | 30.0 |
| Volvo XC40 | ICE | 86.5 | 73.1 | BEV | 16.7 |
| Audi A4 Avant (mild hybrid, shares only) | HEV | 92.4 | 71.0 (82.7 % of windows BEV) | - | - |
| Mercedes GLE 450 (mild hybrid, shares only) | HEV | 91.1 | 49.0 (49.7 % BEV) | - | - |

Within the same car the automation state hardly moves the decision: Audi A6 P(BEV) 71 % on ACC windows vs 76 % on its
manual laps, BMW X5 61 vs 73 %, Mercedes A-Class 49 vs 71 %, Volvo XC40 77 vs 47 %, Mazda 3 55 vs 47 %, Ford S-Max 42
vs 47 %; the BEVs stay BEV in both states (Model 3 78 / 85, Model S 79 / 77, Model X 74 / 62, e-tron 78 / 86, I-Pace
81 / 67, e-Golf 51 / 47). Pooled over the set, ICE recall is 46.4 % on ACC windows and 62.0 % on human windows; BEV
recall 86.3 / 86.2. By campaign (AstaZero, identical laps): Model 3 BEV 80 % right, A8 ICE right, A6 / X5 / A-Class
called BEV; Cherasco: 500X and Polo (human leaders) ICE, XC40 (ACC follower) BEV; Vicolungo: Space Star and MINI ICE,
S-Max 51 % BEV (miss there, ICE over the whole set), e-Golf BEV.

Size (V3 variant primary):

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall SMALL / MID / LARGE % | chance uniform / majority % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V3 variant (primary) | window | 4,540 | 30.2 [15.8, 47.0] | 22.0 [14.0, 28.6] | 82.1 / 10.1 / 3.6 | 33.3 / 39.0 | 47.1 / 41.3; 47.3 / 40.1 |
|  | drive | 630 | 27.1 [10.0, 50.8] | 19.9 [9.7, 28.2] | 92.5 / 9.4 / 1.7 | 33.3 / 45.9 | 53.4 / 45.1; 49.4 / 39.3 |
|  | vehicle | 21 (9 / 6 / 6) | 38.1 [19.0, 57.1] | 18.4 [10.7, 24.2] | 88.9 / 0.0 / 0.0 | 33.3 / 42.9 | 54.4 / 45.6; 56.1 / 47.0 |
|  | vehicle, no borderline | 17 (8 / 5 / 4) | 41.2 [17.6, 64.7] | 19.4 [10.0, 26.2] | 87.5 / 0.0 / 0.0 | 33.3 / 47.1 | - |
| V2 variant | window | 4,540 | 29.6 [20.0, 39.7] | 28.6 [19.1, 35.1] | 37.6 / 46.6 / 10.3 | 33.3 / 39.0 | 41.7 / 34.9; 43.0 / 35.0 |
|  | vehicle | 21 | 42.9 [23.8, 61.9] | 36.4 [19.0, 49.2] | 55.6 / 66.7 / 0.0 | 33.3 / 42.9 | 43.2 / 32.8; 46.2 / 36.0 |
| full model + V2 mask (secondary) | vehicle | 21 | 23.8 [9.4, 42.9] | 13.3 [5.7, 20.7] | 55.6 / 0.0 / 0.0 | 33.3 / 42.9 | 48.0 / 23.2; 51.1 / 26.3 |
| full model + V3 mask (secondary) | vehicle | 21 | 42.9 [23.8, 66.7] | 20.0 [12.8, 26.7] | 100.0 / 0.0 / 0.0 | 33.3 / 42.9 | 49.2 / 25.3; 51.5 / 24.6 |

Named cars, size, V3 variant: SMALL right for Audi A4 Avant (P 56.2 %), Fiat 500X (64.1), MINI Cooper (52.3), Mazda 3
(51.4), Mercedes A-Class (65.2), Mitsubishi Space Star (63.5), VW e-Golf (71.4), Volvo XC40 (71.7); the VW Polo
(3.61 m^2) is the only SMALL miss (LARGE 44.9 %, drive accuracy 0.0). Every MID car is called SMALL: Audi A6 (P(SMALL)
65.1 %), Audi e-tron (51.1, borderline), Ford S-Max (53.2), SsangYong Rexton (62.2), Tesla Model 3 (58.6, two cars).
Every LARGE car is called SMALL: Audi A8 (63.6, borderline), BMW X5 (66.0), Jaguar I-Pace (57.4, borderline),
Mercedes GLE 450 (58.7), Tesla Model S (55.0), Tesla Model X (52.9); P(LARGE) never exceeds 25 % for any car. The
speed regime does not matter: the 10 m/s ZalaZone cars and the 28-32 m/s motorway cars receive the same 50-70 %
P(SMALL).

Caveats: as for ULTra-AV (same campaigns): controller-driven longitudinal behaviour, GNSS speeds and derived
acceleration, one car per model confounded with campaign; the Polo / 500X / Space Star / Rexton / MINI are human-driven
leaders without a lead, so their ICE calls come from speed and acceleration alone; the 150-m lead gate and the GNSS
spacing give a cleaner, more persistent lead than training radar / vision leads; borderline footprints on four cars
(1,087 windows).

### 2.3 microSIM-ACC / EV (`microsim_ev`)

What it is: controlled ACC car-following trials on a straight road (scripted lead dips), 50 Hz GNSS follower / leader
speed and spacing resampled to 10 Hz; 111 windows (1.9 h) from 110 trials >= 60 s: Hyundai Ioniq 5 (99 windows), Polestar
2 (8), Tesla Model 3 (4), all BEV, all ACC-engaged (`DS_MICROSIM_UGA.md`). One class per task present except size
(SMALL 8 / MID 103, both borderline cars).

| task | model | window acc % | drive acc % | vehicle decisions | internal reference (das) |
|---|---|---|---|---|---|
| BEV vs ICE | V2 variant (primary) | 55.0 (BEV recall) | 54.5 (60 of 110 trials) | 3 of 3 BEV: Ioniq 5 P(BEV) 54.4 %, Polestar 2 92.4 %, Model 3 54.1 % | BEV drive recall 78.3, vehicle 74.3 |
| BEV vs ICE | V3 variant | 11.7 | 11.8 | 0 of 3 (all ICE) | BEV drive recall 84.6 |
| BEV vs ICE | full + V2 mask (secondary) | 61.3 | 60.9 | 2 of 3 (Model 3 -> ICE) | BEV drive recall 17.5 |
| BEV vs ICE | full + V3 mask (secondary) | 18.0 | 18.2 | 1 of 3 | BEV drive recall 27.1 |
| size | V3 variant (primary) | 26.1 (SMALL 100.0 / MID 20.4) | 26.4 | Polestar 2 SMALL right (60.4 %); Ioniq 5 MID -> SMALL (47.9 / 38.0 / 14.1); Model 3 MID -> SMALL (47.2 / 40.8 / 12.0) | SMALL 69.1 / MID 25.7 drive recall |
| size | V2 variant | 25.2 (SMALL 0.0 / MID 27.2) | 24.5 | 0 of 3: Ioniq 5 -> SMALL, Polestar 2 -> MID, Model 3 -> SMALL | MID drive recall 14.1 |
| size | full + V2 / V3 mask (secondary) | 23.4 / 9.9 | 23.6 / 10.0 | 0 of 3 / 1 of 3 | - |

Caveats: three cars, 90 % of the windows from one Ioniq 5; scripted protocol (ACC engaged throughout, speed dips to a
stop and back); GNSS speeds interpolated by the authors from 25 Hz; lead distance from antenna-to-antenna spacing minus a
public car length; the two borderline footprints (Ioniq 5 4.83 vs the 4.88 cut, Polestar 2 4.38 vs 4.43) make the size
labels themselves fragile. A BEV-positive sanity probe that passes for the powertrain task with the matched variant and
fails for size.

### 2.4 UGA EV platoon (`uga_ev`)

What it is: 12 scripted platoon runs (lead Lexus UX 200 ICE with a 0 -> 55 mph -> oscillation -> stop profile, followed by a
Ford Mustang Mach-E and two Kia Niro EV; followers on ACC in runs 1-8, manual in 9-12; one- vs two-pedal driving by run),
RaceBox 10 Hz GNSS per car, lead gap from paired positions with a clock-consistency gate (five follower drives lost their
lead group); 326 windows (5.4 h), 4 cars, 48 drives.

| task | model | level | units | accuracy % [95 % CI] | macro-F1 % | recall per class % | chance uniform / majority % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|---|
| BEV vs ICE | V2 variant (primary) | window | 326 (244 / 82) | 42.6 [26.1, 59.6] | 37.6 | BEV 47.5 / ICE 28.0 | 50.0 / 74.8 | 76.7 / 74.1; 75.3 / 73.0 |
|  |  | drive | 48 (36 / 12) | 47.9 [25.0, 70.8] | 42.3 | BEV 52.8 / ICE 33.3 | 50.0 / 75.0 | 90.0 / 89.0; 86.9 / 86.3 |
|  |  | vehicle | 4 (3 / 1) | 50.0 | 33.3 | BEV 66.7 / ICE 0.0 | 50.0 / 75.0 | 91.4 / 90.4; 85.4 / 84.2 |
| BEV vs ICE | V3 variant | vehicle | 4 | 25.0 | 20.0 | BEV 0.0 / ICE 100.0 | 50.0 / 75.0 | 91.0 / 90.1; 86.5 / 85.6 |
| BEV vs ICE | full + V2 / V3 mask (secondary) | vehicle | 4 | 25.0 / 25.0 | 20.0 / 20.0 | BEV 0.0 / ICE 100.0 | 50.0 / 75.0 | 69.3 / 56.8; 65.2 / 49.8 |
| size | V3 variant (primary) | window | 326 (243 / 83 / 0) | 73.9 [39.9, 98.2] | 56.7 [39.2, 60.5] | SMALL 92.2 / MID 20.5 | 33.3 / 74.5 | 47.1 / 41.3; 47.3 / 40.1 |
|  |  | drive | 48 (36 / 12) | 75.0 [25.0, 100.0] | 42.9 | SMALL 100.0 / MID 0.0 | 33.3 / 75.0 | 53.4 / 45.1; 49.4 / 39.3 |
|  |  | vehicle | 4 (3 / 1) | 75.0 | 42.9 | SMALL 100.0 / MID 0.0 | 33.3 / 75.0 | 54.4 / 45.6; 56.1 / 47.0 |
| size | V2 variant | vehicle | 4 | 50.0 | 50.0 | SMALL 33.3 / MID 100.0 | 33.3 / 75.0 | 43.2 / 32.8; 46.2 / 36.0 |
| size | full + V2 / V3 mask (secondary) | vehicle | 4 | 75.0 / 75.0 | 42.9 / 42.9 | SMALL 100.0 / MID 0.0 | 33.3 / 75.0 | 48.0 / 23.2; 49.2 / 25.3 |

Named cars (V2 powertrain / V3 size): Kia Niro EV (two cars): BEV right, P(BEV) 55.3 %, 70.8 % of drives; SMALL right
(61.1 %, every drive). Ford Mustang Mach-E (4.85 m^2, borderline MID): called ICE (P(BEV) 35.8 %, 16.7 % of drives BEV)
and SMALL (55.1 / 36.8 / 8.0). Lexus UX 200 (ICE lead, scripted speed profile, no lead ahead): called BEV (57.8 %) and
SMALL right (59.6 %). On ACC windows (followers, runs 1-8) BEV recall is 44.5 %, on the manual follower runs 53.8 %;
the Mach-E stays ICE-looking in both (P(BEV) 32 vs 44 %).

Caveats: four cars, scripted oscillation protocol on one road, GNSS speeds and derived acceleration, protocol-level
automation flags, lead gaps subject to logger clock offsets (five drives gated), the Mach-E footprint 0.035 m^2 below
the LARGE cut; a sanity probe with a mixed outcome (Niro EV right on both tasks, Mach-E and UX 200 wrong on powertrain).

### 2.5 Yuan et al. 2024 Tesla Model 3 (`tesla_yuan`)

What it is: one 2019 Tesla Model 3 SR+ (MID 4.57 m^2, BEV; a training platform of `split_platform`), 106 real-world
trips in Changchun (urban stop-and-go, mean speed 9.4 m/s against 17.5 in training), 20 Hz CAN speed decimated to 10 Hz,
derived acceleration, Autopilot use unrecorded (`das_enabled` = 0 assumed); 2,619 windows (43.7 h) (`DS_TESLA_YUAN.md`).
Speed-only set: V3 variant primary.

| task | model | window | drive (106 trips) | vehicle | internal reference of the same quantity (human-only / all) |
|---|---|---|---|---|---|
| BEV vs ICE | V3 variant (primary) | BEV recall 62.3 % | 80.2 % (85 trips) | BEV, P(BEV) 60.8 % | V3 BEV recall: window 67.7 / 68.2, drive 77.2 / 78.9, vehicle 74.7 / 79.6 |
| BEV vs ICE | full + V3 mask (secondary) | 10.9 % | 0.0 % | ICE | full + V3 mask BEV drive recall 15.0 / 16.8 |
| size | V3 variant (primary) | MID recall 63.7 % | 78.3 % (83 trips) | MID, P 18.8 / 50.0 / 31.2 % | V3 MID recall: window 34.2 / 36.7, drive 43.6 / 37.7, vehicle 41.5 / 32.3 |
| size | full + V3 mask (secondary) | 20.0 % | 7.5 % | SMALL | full + V3 mask MID drive recall 4.8 |

Reading: the matched variant calls the car BEV in 4 of 5 trips, at the level of its own BEV recall on human-driven
held-out windows (77.2 % of drives), and MID in 78.3 % of the trips, twice its internal MID recall; both are single-car
positives from a training platform and a different sensor chain, region and driver pool, not a generalisation result.
Caveats: `das_enabled` = 0 is an assumption (Autopilot possible on the expressway trips); urban distribution shift;
derived acceleration from a held 20 Hz speed; one car, no interval.

### 2.6 OpenACC one-vehicle campaign (`openacc_500x`)

What it is: one Fiat 500X diesel (SMALL 3.97 m^2, ICE) driven manually by 20 volunteers on 438 trips (CAN / OBD speed only,
interpolated by JRC to 10 Hz; engine speed and gear never read); 13,523 windows (225.4 h) from 429 trips after the speed
cleaning of `DS_OPENACC.md`.

| task | model | window | drive (429 trips) | vehicle | internal reference (human-only / all) |
|---|---|---|---|---|---|
| BEV vs ICE | V3 variant (primary) | ICE recall 87.9 % | 99.1 % (425 trips) | ICE, P(ICE) 81.8 % | V3 ICE recall: window 81.6 / 81.5, drive 93.0 / 96.3, vehicle 99.2 / 98.0 |
| BEV vs ICE | full + V3 mask (secondary) | 90.4 % | 100.0 % | ICE | full + V3 mask ICE drive recall 93.1 (BEV 16.8: the masked model says ICE almost regardless) |
| size | V3 variant (primary) | SMALL recall 57.3 % | 46.2 % (198 trips) | SMALL, P 47.2 / 38.5 / 14.3 % | V3 SMALL recall: window 57.7 / 58.5, drive 66.3 / 73.4, vehicle 63.7 / 82.4 |
| size | full + V3 mask (secondary) | 92.2 % | 98.4 % | SMALL | full + V3 mask SMALL drive recall 98.2 (it says SMALL for everything) |

Reading: the cleanest ICE negative control of the campaign, 20 drivers on 225 h, and the speed-only variant calls 99.1 %
of the trips ICE (a human-driven ICE car in traffic is what the ICE class looks like to it). The SMALL call is weaker
than internal (46.2 % of trips against 66.3 % of our human-driven drives) but right at vehicle level. Caveats: one car,
OBD speed with 1/16 km/h quantisation and interpolation holds (cleaned), derived acceleration, no lateral channels.

### 2.7 VED (`ved`)

What it is: 384 privately owned cars in Ann Arbor (264 ICE, 93 HEV, 24 PHEV, 3 BEV = 2013 Nissan Leaf, a held-out test
model of `split_platform`), make and model withheld (the "models" are powertrain x 500-lb weight bins), ~1 Hz OBD speed
homogenised to 1 Hz / 1 km/h and upsampled to 10 Hz, derived acceleration with a 1-s window; 59,563 windows (992.7 h),
25,580 trips, at most 200 windows per car (`DS_VED.md`). Speed-only: V3 variant primary.

Powertrain:

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall BEV / ICE % | chance uniform / majority % | internal reference acc / F1 (all; human) |
|---|---|---|---|---|---|---|---|
| V3 variant (primary) | window | 39,259 (431 / 38,828) | 68.8 [67.4, 70.1] | 41.3 [40.7, 42.0] | 17.4 / 69.3 | 50.0 / 98.9 | 77.0 / 74.6; 78.0 / 73.0 |
|  | drive | 15,770 (317 / 15,453) | 74.6 [72.3, 76.6] | 44.0 [42.9, 45.3] | 16.4 / 75.8 | 50.0 / 98.0 | 89.5 / 88.6; 87.6 / 85.9 |
|  | vehicle | 267 (3 / 264) | 96.6 [94.4, 98.5] | 49.1 [48.6, 49.6] | 0.0 / 97.7 | 50.0 / 98.9 | 91.0 / 90.1; 89.8 / 88.6 |
| full model + V3 mask (secondary) | window | 39,259 | 95.1 [93.7, 96.2] | 50.3 [49.0, 51.6] | 7.2 / 96.1 | 50.0 / 98.9 | 69.7 / 59.4; 69.8 / 54.5 |
|  | vehicle | 267 | 98.1 [96.6, 99.6] | 49.5 [49.0, 49.8] | 0.0 / 99.2 | 50.0 / 98.9 | 65.2 / 49.8; 63.6 / 50.1 |

Per true powertrain (V3 variant): share of windows / drives / vehicles called BEV: ICE 30.7 / 24.2 / 2.3 % (6 of 264 cars),
HEV 24.3 / 18.2 / 1.1 % (1 of 93), PHEV 22.6 / 18.1 / 0.0 % (0 of 24), BEV (Leaf) 17.4 / 16.4 / 0.0 % (0 of 3); mean
P(BEV) 0.34 (ICE), 0.29 (HEV), 0.28 (PHEV), 0.25 (Leaf). The ICE weight bins are all ICE at model level (P(ICE) 63.6-75.6 %,
drive accuracy 71.6-80.8 %, 100.0 for the single 6,500-lb car); the Leaf gets a lower BEV probability than the ICE cars. Internal counterpart: on our own
held-out Nissan Leaf (15 cars, 945 windows) the V3 variant calls 18 % of the drives BEV and the model ICE (P(BEV)
37 %), V2 21 %, V1 16 %, and only the full 20-channel model gets it (64 % of drives, model-level BEV). The VED result
therefore reproduces a known weakness of the speed-only variant on exactly this model, on top of the 1 Hz upsampling
(section 3.3: the VED-like resampling alone cuts the internal V3 BEV / ICE separation to 56.5 / 70.4 drive recall).

Size: only the three Leafs carry a label (SMALL, 4.149 m^2): V3 window recall 60.3 % [54.8, 63.5], drive 62.5 % [58.7,
65.9] (317 trips), 3 of 3 cars SMALL (P 46.5 / 38.9 / 14.6 %); our V3 reference for SMALL human-driven drives is 66.3 %.
Weight-bin ordinal check over all 384 cars (predicted shares SMALL / MID / LARGE of the V3 size variant): 2,500 lb
45.4 / 46.0 / 8.6 %, 3,000 lb 46.9 / 42.3 / 10.8, 3,500 lb 46.2 / 45.6 / 8.2, 4,000 lb 52.1 / 41.4 / 6.5, 4,500 lb
40.9 / 48.3 / 10.8, 5,000 lb 49.2 / 45.9 / 4.9, 5,500 lb 43.5 / 52.0 / 4.5, 6,000 lb 45.5 / 47.0 / 7.6; the ordinal score
1 x P(SMALL) + 2 x P(MID) + 3 x P(LARGE) stays between 1.73 and 1.84 and its vehicle-level Spearman correlation with the
weight bin is -0.125 (n = 368): no size information survives in VED's channels. (The full model under the V3 mask calls
98.1 % of the Leaf windows SMALL, as it calls everything SMALL.)

Caveats: 1 Hz upsampled OBD speed (the only measured channel), class-correlated logging artefacts homogenised but not
removable (EV loggers poll faster), 3 BEV cars of one 2013 model against 264 ICE cars, weight bins instead of models,
`das_enabled` = 0 assumed, hybrids unlabelled for the task.

### 2.8 A2D2 Audi Q7 e-tron (`a2d2`)

What it is: the three sensor-fusion bus-signal sequences of A2D2 (one Audi Q7 e-tron quattro, diesel PHEV, LARGE
5.045 m^2, manual urban driving, 36.5 min -> 34 windows, 0.57 h): speed, derived acceleration, steering angle (and
derived rate), IMU yaw rate, curvature and lateral acceleration by the lab definitions, pedal flags, standstill, das = 0;
the lead group and the torque channel absent in the training dropout encoding; `steer_pressed` unknown (NaN -> the
standardised mean, the only out-of-distribution input). A single-car LARGE sanity probe for the full 20-channel size
model; the powertrain (PHEV) is outside the task (`DS_A2D2.md`).

| task | model | windows (34) | drives (3) | vehicle | decision (P SMALL / MID / LARGE %) | internal reference LARGE recall, drive (all / human) |
|---|---|---|---|---|---|---|
| size | full 20-ch model (primary) | 0.0 % LARGE | 0 of 3 | miss | SMALL 85.2 / 9.2 / 5.6 | 58.6 / 60.3 |
| size | V1 variant (lateral + speed) | 2.9 % | 0 of 3 | miss | SMALL 68.0 / 19.0 / 13.0 | 62.5 / 66.3 |
| size | full + V1 mask (secondary) | 0.0 % | 0 of 3 | miss | SMALL | 46.6 / 60.3 |
| size | V3 variant (speed-only) | 14.7 % | 0 of 3 | miss | MID 36.0 / 43.0 / 21.0 | 23.9 / 30.2 |
| size | full + V3 mask (secondary) | 0.0 % | 0 of 3 | miss | SMALL | 0.4 / 0.0 |
| BEV vs ICE (PHEV, shares only) | full 20-ch model | ICE 100.0 % of windows | - | - | - | - |
|  | V1 variant / full + V1 mask | ICE 58.8 / 64.7 % | - | - | - | - |
|  | V3 variant / full + V3 mask | ICE 85.3 / 91.2 % | - | - | - | - |

Reading (triage rule: artefacts first, then mechanism, then the user decides): the adapter checks found no sign, scale or
range problem (`DS_A2D2.md`, sections 2, 5 and 7), and `DS_A2D2.md` section 7 shows that the decision is fragile where the
data are blind (steer_pressed := 1 flips it to MID 96 %, V3das to LARGE). The internal counterpart is informative: the
full model misses LARGE sedans and crossover SUVs of unseen makes on our own data too (Tesla Model S LARGE recall 6 %,
Genesis G80 / G90 0 %, VW Atlas 0 %, Toyota Sienna 45 %) while getting pickups and large three-row SUVs (F-150 Lightning
95 %, Ranger 90 %, Yukon 100 %, GV80 62 %, Carnival 77 %): LARGE is learned as truck- or minivan-like, and a 5.0 m^2
European SUV at 5 m/s median speed is not that. Caveats: one car, one region, 36 min, low-speed manual driving, CC
BY-ND data (internal evaluation only).

### 2.9 commaCarSegments, verification sample (`ccs_sample`)

What it is: the five verification segments decoded by `ccs_decode.py` (all 20 channels; radar lead on the RAV4 only,
torque scale on two cars, yaw steering-derived) -> 4 windows, one per car (the Ioniq 5 minute is 83 % stationary and
dropped by the moving rule); `das_enabled` = cruise or controls-allowed. A pipeline check, not a result: one 60-s window
per car, full model only.

| car | footprint | size truth -> decision (P S / M / L %) | powertrain truth -> decision (P BEV %) | DAS % |
|---|---|---|---|---|
| Toyota RAV4 TSS2 (radar lead) | 4.34 | SMALL -> SMALL (86.3 / 11.8 / 1.9) | ICE -> BEV (83.0) miss | 99.8 |
| Tesla Model 3 (2025 port) | 4.57 | MID -> MID (21.2 / 65.1 / 13.7) | BEV -> BEV (98.6) | 0.0 |
| Chevrolet Bolt EUV | 3.99 | SMALL -> MID (14.9 / 70.8 / 14.4) miss | BEV -> BEV (98.6) | 35.3 |
| Rivian R1T / R1S | 5.58 | LARGE -> MID (23.9 / 42.9 / 33.2) miss | BEV -> BEV (87.2) | 99.8 |

Size 2 of 4, powertrain 3 of 4 (internal full-model references: 63.5 / 93.9 % vehicle accuracy). The RAV4 window is an
ACC-engaged highway minute with a radar lead and no driver pedal or steering input, called BEV with 83 %: the same pattern
as the ACC-following ICE cars of OpenACC, here with the full channel set and a CAN `aEgo` (so not a derived-acceleration
artefact). All four are training platforms (device-disjoint by construction of the selection).

### 2.10 commaSteeringControl (`csc`)

What it is: comma's 2023 release of 60-s openpilot segments in which the lateral controller was active, one zip archive
per 2023 platform; the 36 archives the owner downloaded (9 BEV, 27 ICE) were read directly from the zips, 1,500
segments per archive in a seeded random order (every eligible segment for the five smaller archives) -> 51,462 windows
(857.7 h), one window per segment, 25 car models; no device or route ids, so one archive = one `vehicle_id` (36
pseudo-vehicles; the EPS firmware strings give 1-17 versions per archive as a lower bound on distinct cars) and drive =
segment = window. Channels (`DS_CSC.md`): CAN `vEgo` / `aEgo`, steering angle (rate = 10 Hz central difference),
yaw / curvature / lateral acceleration from the localizer lateral acceleration (sign flipped to the lab's left-positive
convention), `steer_pressed`, standstill = v < 0.3 m/s, `das_enabled` = lateral controller active (92 % of the samples;
every window is automation-engaged; the longitudinal state is not logged, but on openpilot a lateral-active segment
implies an engaged cruise / longitudinal controller - stock ACC on pcmCruise cars or openpilot's own - except during
driver gas overrides); lead, torque and pedal groups absent with
`lead_valid` = 0, i.e. exactly the V1 channel list. Highway-biased (mean speed 24.6 m/s against 17.6 in our windows,
1.2 % of the samples below 0.3 m/s). Labels through the opendbc `MIGRATION` table to the lab platform table (footprint,
model) and the drives index (stratum: 26 archives / 18 models seen in training or validation, 10 archives / 7 models
held-out test models of `split_platform`); powertrain from the 2023 archive identity (hybrids were separate archives,
none downloaded), so six archives whose merged current platform the lab labelled HEV / MIXED (Accord, Tucson, Camry
2018 / 2021, Corolla TSS2, RAV4 2022) are ICE here, and two more (Highlander 2020, RAV4 2019) fold onto ICE-majority
mixes. Labels: 11,963 BEV
windows (9 archives, 8 models) / 39,499 ICE (27, 17); size 23,964 SMALL (17 archives, 10 models) / 18,498 MID (13, 9) /
9,000 LARGE (6, 6); borderline Camry 2018 / 2021 (4.47), Pilot (4.92), Ioniq 5 (4.83).

Size (V1 variant primary; window = segment = drive, vehicle = archive):

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall SMALL / MID / LARGE % | chance uniform / majority % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V1 variant (primary, matched inputs) | window | 51,462 | 48.4 [41.5, 56.1] | 44.1 [37.8, 48.6] | 68.1 / 29.3 / 35.1 | 33.3 / 46.6 | 62.0 / 55.0; 61.2 / 49.8 |
|  | vehicle (archive) | 36 (17 / 13 / 6) | 63.9 [50.0, 80.6] | 62.3 [42.9, 78.1] | 88.2 / 30.8 / 66.7 | 33.3 / 47.2 | 62.6 / 58.0; 62.6 / 54.9 |
|  | vehicle, no borderline | 32 (17 / 10 / 5) | 65.6 [50.0, 81.2] | 60.9 [38.1, 79.3] | 88.2 / 30.0 / 60.0 | 33.3 / 53.1 | - |
|  | model | 25 (10 / 9 / 6) | 64.0 [50.0, 76.5] | 64.3 [45.4, 77.5] | 90.0 / 33.3 / 66.7 | 33.3 / 40.0 | - |
| V3 variant (speed-only) | window | 51,462 | 47.0 [40.4, 53.8] | 42.5 [36.1, 47.2] | 65.3 / 30.1 / 32.8 | 33.3 / 46.6 | 47.1 / 41.3; 47.3 / 40.1 |
|  | vehicle | 36 | 55.6 [38.9, 72.2] | 44.9 [24.5, 57.4] | 100.0 / 0.0 / 50.0 | 33.3 / 47.2 | 54.4 / 45.6; 56.1 / 47.0 |
|  | model | 25 | 52.0 [42.1, 64.7] | 43.1 [22.2, 56.4] | 100.0 / 0.0 / 50.0 | 33.3 / 40.0 | - |
| full model + V1 mask (secondary) | window | 51,462 | 47.6 [33.8, 62.9] | 29.7 [23.9, 34.7] | 94.8 / 4.9 / 9.6 | 33.3 / 46.6 | 63.2 / 51.2; 63.4 / 42.3 |
|  | vehicle | 36 | 47.2 [30.6, 63.9] | 21.4 [15.6, 26.0] | 100.0 / 0.0 / 0.0 | 33.3 / 47.2 | 60.8 / 50.8; 55.3 / 36.6 |

Powertrain (V3 variant primary, see 1.3):

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall BEV / ICE % | chance uniform / majority % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V3 variant (primary) | window | 51,462 | 75.4 [72.2, 78.5] | 69.0 [61.3, 73.2] | 63.9 / 78.9 | 50.0 / 76.8 | 77.0 / 74.6; 75.1 / 73.4 |
|  | vehicle (archive) | 36 (9 / 27) | 97.2 [91.7, 100.0] | 96.1 [87.4, 100.0] | 88.9 / 100.0 | 50.0 / 75.0 | 91.0 / 90.1; 86.5 / 85.6 |
|  | model | 25 (8 / 17) | 96.0 [93.8, 100.0] | 95.2 [90.9, 100.0] | 87.5 / 100.0 | 50.0 / 68.0 | - |
| V1 variant (the matched input list) | window | 51,462 | 79.4 [69.0, 88.2] | 61.7 [55.3, 67.1] | 24.4 / 96.1 | 50.0 / 76.8 | 70.0 / 67.9; 69.2 / 68.0 |
|  | vehicle | 36 | 75.0 [61.1, 86.1] | 42.9 [37.9, 46.3] | 0.0 / 100.0 | 50.0 / 75.0 | 75.8 / 74.6; 77.6 / 76.8 |
|  | model | 25 | 68.0 [60.0, 82.4] | 40.5 [37.5, 45.2] | 0.0 / 100.0 | 50.0 / 68.0 | - |
| full model + V1 mask (secondary) | window | 51,462 | 77.8 [66.5, 86.7] | 57.3 [53.5, 59.4] | 18.3 / 95.8 | 50.0 / 76.8 | 68.9 / 62.4; 73.5 / 67.1 |
|  | vehicle | 36 | 75.0 [61.1, 86.1] | 42.9 [37.9, 46.3] | 0.0 / 100.0 | 50.0 / 75.0 | 69.3 / 61.2; 76.6 / 71.8 |

Per archive (mean probability over the archive's windows -> decision; sorted by class and footprint; `RESULTS_TABLES.md`
has every run):

| archive (2023 platform) | model | footprint | stratum | win | size truth -> V1 decision (P S, M, L %) | V3 size | pt truth -> V3 decision (P BEV %) | V1 (P BEV %) |
|---|---|---|---|---|---|---|---|---|
| CHEVROLET BOLT EUV 2022 | Chevrolet Bolt EUV | 3.99 | seen | 1,500 | SMALL -> SMALL (54, 30, 16) OK | SMALL OK | BEV -> BEV (66) OK | ICE (35) MISS |
| HYUNDAI KONA ELECTRIC 2019 | Hyundai Kona | 4.02 | held-out | 1,500 | SMALL -> SMALL (56, 35, 10) OK | SMALL OK | BEV -> BEV (55) OK | ICE (35) MISS |
| TOYOTA COROLLA TSS2 2019 | Toyota Corolla | 4.09 | held-out | 1,500 | SMALL -> SMALL (56, 35, 10) OK | SMALL OK | ICE -> ICE (20) OK | ICE (8) OK |
| TOYOTA RAV4 2017 | Toyota RAV4 | 4.13 | seen | 1,500 | SMALL -> SMALL (47, 31, 22) OK | SMALL OK | ICE -> ICE (22) OK | ICE (9) OK |
| TOYOTA COROLLA 2017 | Toyota Corolla | 4.14 | held-out | 1,500 | SMALL -> SMALL (44, 30, 26) OK | SMALL OK | ICE -> ICE (37) OK | ICE (8) OK |
| NISSAN LEAF 2018 | Nissan Leaf | 4.16 | held-out | 1,500 | SMALL -> SMALL (49, 32, 20) OK | SMALL OK | BEV -> BEV (57) OK | ICE (28) MISS |
| NISSAN LEAF 2018 Instrument Cluster | Nissan Leaf | 4.16 | held-out | 1,467 | SMALL -> SMALL (57, 31, 13) OK | SMALL OK | BEV -> BEV (53) OK | ICE (18) MISS |
| HONDA CIVIC 2016 | Honda Civic | 4.20 | seen | 1,500 | SMALL -> SMALL (59, 35, 6) OK | SMALL OK | ICE -> ICE (28) OK | ICE (18) OK |
| HONDA CIVIC BOSCH 2019 | Honda Civic | 4.20 | seen | 1,500 | SMALL -> SMALL (57, 36, 6) OK | SMALL OK | ICE -> ICE (27) OK | ICE (20) OK |
| HONDA CIVIC 2022 | Honda Civic | 4.21 | seen | 1,500 | SMALL -> SMALL (58, 37, 5) OK | SMALL OK | ICE -> ICE (29) OK | ICE (17) OK |
| HYUNDAI IONIQ ELECTRIC LIMITED 2019 | Hyundai Ioniq | 4.22 | held-out | 497 | SMALL -> MID (40, 51, 9) MISS | SMALL OK | BEV -> ICE (32) MISS | ICE (25) MISS |
| HONDA CR V 2016 | Honda CR-V | 4.23 | seen | 1,000 | SMALL -> LARGE (40, 13, 48) MISS | SMALL OK | ICE -> ICE (27) OK | ICE (12) OK |
| KIA NIRO EV 2020 | Kia Niro | 4.27 | seen | 1,500 | SMALL -> SMALL (56, 35, 9) OK | SMALL OK | BEV -> BEV (60) OK | ICE (38) MISS |
| HONDA CR V 2017 | Honda CR-V | 4.28 | seen | 1,500 | SMALL -> SMALL (54, 29, 17) OK | SMALL OK | ICE -> ICE (25) OK | ICE (11) OK |
| HYUNDAI ELANTRA 2021 | Hyundai Elantra | 4.31 | seen | 1,500 | SMALL -> SMALL (56, 36, 7) OK | SMALL OK | ICE -> ICE (31) OK | ICE (13) OK |
| TOYOTA RAV4 2019 | Toyota RAV4 | 4.34 | seen | 1,500 | SMALL -> SMALL (55, 33, 12) OK | SMALL OK | ICE -> ICE (18) OK | ICE (9) OK |
| TOYOTA RAV4 2022 | Toyota RAV4 | 4.34 | seen | 1,500 | SMALL -> SMALL (57, 31, 12) OK | SMALL OK | ICE -> ICE (16) OK | ICE (7) OK |
| TOYOTA CAMRY 2018 | Toyota Camry | 4.47 border | seen | 1,500 | MID -> SMALL (57, 34, 9) MISS | SMALL MISS | ICE -> ICE (22) OK | ICE (11) OK |
| TOYOTA CAMRY 2021 | Toyota Camry | 4.47 border | seen | 1,500 | MID -> SMALL (58, 35, 7) MISS | SMALL MISS | ICE -> ICE (16) OK | ICE (11) OK |
| HYUNDAI TUCSON 4TH GEN | Hyundai Tucson | 4.53 | seen | 999 | MID -> MID (39, 45, 16) OK | SMALL MISS | ICE -> ICE (25) OK | ICE (8) OK |
| HYUNDAI SANTA FE 2019 | Hyundai Santa Fe | 4.53 | held-out | 1,500 | MID -> SMALL (51, 28, 20) MISS | SMALL MISS | ICE -> ICE (31) OK | ICE (12) OK |
| HONDA ACCORD 2018 | Honda Accord | 4.54 | seen | 1,500 | MID -> SMALL (52, 35, 13) MISS | SMALL MISS | ICE -> ICE (24) OK | ICE (9) OK |
| HYUNDAI SONATA 2019 | Hyundai Sonata | 4.56 | seen | 1,500 | MID -> SMALL (58, 35, 7) MISS | SMALL MISS | ICE -> ICE (32) OK | ICE (22) OK |
| HYUNDAI SANTA FE 2022 | Hyundai Santa Fe | 4.57 | held-out | 1,500 | MID -> SMALL (50, 29, 21) MISS | SMALL MISS | ICE -> ICE (28) OK | ICE (10) OK |
| TOYOTA HIGHLANDER 2017 | Toyota Highlander | 4.60 | seen | 1,500 | MID -> MID (31, 35, 34) OK | SMALL MISS | ICE -> ICE (30) OK | ICE (4) OK |
| HYUNDAI SONATA 2020 | Hyundai Sonata | 4.62 | seen | 1,500 | MID -> SMALL (48, 29, 22) MISS | SMALL MISS | ICE -> ICE (32) OK | ICE (11) OK |
| TOYOTA HIGHLANDER 2020 | Toyota Highlander | 4.68 | seen | 1,500 | MID -> SMALL (52, 37, 11) MISS | SMALL MISS | ICE -> ICE (21) OK | ICE (9) OK |
| GENESIS GV60 ELECTRIC 1ST GEN | Genesis GV60 | 4.73 | held-out | 999 | MID -> MID (38, 43, 19) OK | SMALL MISS | BEV -> BEV (58) OK | ICE (19) MISS |
| KIA EV6 2022 | Kia EV6 | 4.74 | seen | 1,500 | MID -> SMALL (55, 36, 10) MISS | SMALL MISS | BEV -> BEV (65) OK | ICE (49) MISS |
| HYUNDAI IONIQ 5 2022 | Hyundai Ioniq 5 | 4.83 border | seen | 1,500 | MID -> MID (37, 43, 20) OK | SMALL MISS | BEV -> BEV (66) OK | ICE (24) MISS |
| HONDA PILOT 2017 | Honda Pilot | 4.91 border | seen | 1,500 | LARGE -> LARGE (28, 20, 52) OK | LARGE OK | ICE -> ICE (22) OK | ICE (8) OK |
| HYUNDAI PALISADE 2020 | Hyundai Palisade | 4.96 | seen | 1,500 | LARGE -> SMALL (44, 36, 19) MISS | SMALL MISS | ICE -> ICE (33) OK | ICE (9) OK |
| HONDA ODYSSEY 2018 | Honda Odyssey | 5.14 | seen | 1,500 | LARGE -> SMALL (43, 34, 23) MISS | SMALL MISS | ICE -> ICE (37) OK | ICE (9) OK |
| TOYOTA SIENNA 2018 | Toyota Sienna | 5.21 | held-out | 1,500 | LARGE -> LARGE (34, 27, 39) OK | SMALL MISS | ICE -> ICE (40) OK | ICE (7) OK |
| CHEVROLET SILVERADO 1500 2020 | Chevrolet Silverado 1500 | 6.65 | seen | 1,500 | LARGE -> LARGE (28, 34, 38) OK | LARGE OK | ICE -> ICE (34) OK | ICE (14) OK |
| RAM 1500 5TH GEN | RAM 1500 | 6.73 | seen | 1,500 | LARGE -> LARGE (36, 25, 39) OK | LARGE OK | ICE -> ICE (33) OK | ICE (13) OK |

Pooled per stratum (archive = pseudo-vehicle; model = car model; the full table with the window-level recalls is in
`RESULTS_TABLES.md`):

| task, run | stratum | archives | archive acc / F1 % | archives right per class | models right (acc %) | window acc / F1 % |
|---|---|---|---|---|---|---|
| size, V1 | all | 36 | 63.9 / 62.3 | SMALL 15/17, MID 4/13, LARGE 4/6 | 16/25 (64.0) | 48.4 / 44.1 |
|  | seen | 26 | 61.5 / 59.8 | SMALL 10/11, MID 3/10, LARGE 3/5 | 11/18 (61.1) | 48.1 / 44.7 |
|  | held-out | 10 | 70.0 / 72.3 | SMALL 5/6, MID 1/3, LARGE 1/1 | 5/7 (71.4) | 49.1 / 41.8 |
| size, V3 | all | 36 | 55.6 / 44.9 | SMALL 17/17, MID 0/13, LARGE 3/6 | 13/25 (52.0) | 47.0 / 42.5 |
|  | seen | 26 | 53.8 / 46.6 | SMALL 11/11, MID 0/10, LARGE 3/5 | 9/18 (50.0) | 45.2 / 42.9 |
|  | held-out | 10 | 60.0 / 25.0 | SMALL 6/6, MID 0/3, LARGE 0/1 | 4/7 (57.1) | 52.1 / 38.2 |
| powertrain, V3 | all | 36 | 97.2 / 96.1 | BEV 8/9, ICE 27/27 | 24/25 (96.0) | 75.4 / 69.0 |
|  | seen | 26 | 100.0 / 100.0 | BEV 4/4, ICE 22/22 | 18/18 (100.0) | 78.6 / 68.7 |
|  | held-out | 10 | 90.0 / 89.9 | BEV 4/5, ICE 5/5 | 6/7 (85.7) | 66.6 / 65.7 |
| powertrain, V1 | all | 36 | 75.0 / 42.9 | BEV 0/9, ICE 27/27 | 17/25 (68.0) | 79.4 / 61.7 |
|  | seen | 26 | 84.6 / 45.8 | BEV 0/4, ICE 22/22 | 14/18 (77.8) | 85.6 / 66.4 |
|  | held-out | 10 | 50.0 / 33.3 | BEV 0/5, ICE 5/5 | 3/7 (42.9) | 62.1 / 51.4 |

The seven held-out models next to their internal result on our own test drives (`internal_per_model.md`, same variants;
P of the true class in brackets):

| held-out model (archives) | label | CSC size, V1 | internal size, V1 (drive acc %) | CSC powertrain, V3 | internal powertrain, V3 (drive acc %) |
|---|---|---|---|---|---|
| Genesis GV60 | MID / BEV | MID (0.43) OK | SMALL (0.44) miss, 84 | BEV (0.58) OK | BEV (0.70) OK, 97 |
| Hyundai Ioniq Electric | SMALL / BEV | MID (0.40) miss | SMALL (0.58) OK, 63 | ICE (0.32) miss | BEV (0.59) OK, 90 |
| Hyundai Kona Electric | SMALL / BEV | SMALL (0.56) OK | SMALL (0.69) OK, 91 | BEV (0.55) OK | not comparable (the lab's held-out `Hyundai Kona` model is mostly ICE Konas, labelled ICE at model level) |
| Hyundai Santa Fe (2019, 2022) | MID / ICE | SMALL (0.28, 0.29) miss, miss | MID (0.51) OK, 66 | ICE (0.69, 0.72) OK, OK | ICE (0.73) OK, 90 |
| Nissan Leaf (2018, Instrument Cluster) | SMALL / BEV | SMALL (0.49, 0.57) OK, OK | SMALL (0.69) OK, 95 | BEV (0.57, 0.53) OK, OK | ICE (0.37) miss, 18 |
| Toyota Corolla (2017, TSS2 2019) | SMALL / ICE | SMALL (0.44, 0.56) OK, OK | SMALL (0.65) OK, 93 | ICE (0.63, 0.80) OK, OK | ICE (0.82) OK, 97 |
| Toyota Sienna | LARGE / ICE | LARGE (0.39) OK | LARGE (0.56) OK, 55 | ICE (0.60) OK | ICE (0.66) OK, 100 |

Reading:

* **Size with steering present transfers at the level of our own held-out test.** The V1 archive-level accuracy and
  macro-F1 (63.9 / 62.3) are at or above the internal V1 vehicle-level reference (62.6 / 58.0; automation-engaged
  subset 62.6 / 54.9), with the same class profile: SMALL nearly always right (15 of 17 archives), LARGE about two
  thirds (4 of 6), MID weak (4 of 13 archives; internally 43 % vehicle-level MID recall). Model level 16 of 25 against
  25 of 41 internally. The window-level numbers are lower than internally (48.4 / 44.1 against 62.0 / 55.0) because
  every CSC window is an engaged highway minute with little of the low-speed manoeuvring that a 60-s window of a mixed
  drive contains; the per-archive mean probabilities separate less sharply than the decisions suggest (P(true) 0.35-0.59
  for the hits, 0.19-0.40 for the misses, overlapping among the MID archives).
* **LARGE on unknown devices: pickups and the Sienna.** Silverado 1500 (P(LARGE) 0.38), RAM 1500 (0.39), Pilot (0.52,
  borderline) and the held-out Toyota Sienna (0.39) are LARGE; the Odyssey (0.23) and the Palisade (0.19) are SMALL, as
  the full model misses crossover SUVs internally (3.2). The speed-only variant on the same windows gets the two
  pickups and the Pilot but not the Sienna (14.7 % of its windows LARGE), so the lateral channels add the minivan and
  all four MID hits, i.e. what phase 1 could not test.
* **MID remains the weak class**: GV60 (0.43, held-out), Ioniq 5 (0.43, borderline), Tucson (0.45) and Highlander 2017
  (0.35) are MID; Accord, Camry 2018 / 2021, Sonata 2019 / 2020, Santa Fe 2019 / 2022, Highlander 2020 and EV6 are
  SMALL with P(MID) 0.28-0.37. The MID archives are sedans and compact crossovers of 4.47-4.83 m^2 driven on highways
  with the lateral controller active, where the steering geometry at a given speed differs little from a SMALL car's;
  the lateral signal that separates MID internally (56 % recall on human-only drives) needs manoeuvres that these
  segments rarely contain.
* **Two SMALL misses are input quirks rather than size errors.** HONDA CR-V 2016 is LARGE with 0.48 under V1 but SMALL
  under V3 and under the masked full model: `steeringPressed` is set in 3 of its 1,000 segments (DS_CSC.md 5), a flag
  state no SMALL training car shows for 60-s stretches. The Ioniq Electric (497 windows, one EPS firmware, P(MID) 0.51)
  is the smallest archive.
* **Powertrain: the speed-only signature transfers across 25 models with a CAN acceleration.** V3 archive accuracy 97.2 %
  (35 of 36; model level 24 of 25) against 91.0 % internally, with firm margins (P(BEV) 0.53-0.66 on the BEV archives,
  0.16-0.40 on the ICE archives) built from 500-1,500 windows whose individual accuracy is only 75.4 %. The held-out
  stratum is 9 of 10. Both Leaf archives are BEV (0.57 / 0.53) although the internal V3 misses our own Leaf (18 % of
  its drives): the CSC Leaf windows are engaged highway minutes at 21-22 m/s, the lab's Leaf drives are mixed urban
  driving; the Ioniq Electric is the one miss (0.32). Every ICE archive is ICE, and these are engaged highway minutes: on
  openpilot a lateral-active segment implies that the longitudinal automation (stock ACC on pcmCruise cars, or
  openpilot's own longitudinal control) is engaged too, except during driver gas overrides, so the 27 ICE archives
  are automation-engaged ICE cars read correctly. The ICE misses of phase 1 (A6, X5, Mazda 3, XC40: ACC car-following
  on test tracks with GNSS speeds, a different regime from highway cruising) do not recur when the acceleration is
  openpilot's own `aEgo`, which is what the derived-acceleration diagnostic of 3.3 predicted; the test-track
  car-following regime itself is not re-tested here.
* **The matched V1 variant fails on powertrain (BEV 0 of 9, window BEV recall 24.4 %)**, and the full model under the V1
  mask does the same (0 of 9; internal full + V1 mask BEV vehicle recall 31 %): with the lateral channels and the
  automation flag present and the pedal, torque and lead groups absent, the model reads the steering geometry of a
  platform rather than its deceleration texture, and on a new fleet of engaged highway segments that reads as ICE.
  Internally V1 already was the weakest powertrain variant (74.6 drive accuracy, BEV recall 70.6); externally the loss
  is total.
* **Stratum**: the held-out archives score as well as or better than the seen ones (size 70.0 against 61.5 archive
  accuracy; powertrain 90.0 against 100.0), so the possible overlap between comma's `seen` segments and the lab's
  training routes does not carry the result.

Caveats: no device ids, so one archive is one pseudo-vehicle (the bootstrap resamples 36 archives; the EPS-firmware
count bounds the number of cars per archive from below at 1-17) and an overlap of the `seen` archives with the lab's
training routes cannot be excluded (same openpilot fleet, no route key: read `seen` as an upper bound and `held-out` as
the clean number); every window is an engaged highway minute (das 92 %, mean speed 24.6 m/s; the lab's `das` subset is
the reference, the human-only protocol does not apply; the longitudinal state is inferred from the lateral-active
flag, not logged); derived lateral channels
(rate from the 10 Hz angle, yaw and curvature from the localizer lateral acceleration, `das_enabled` from the
lateral-active flag; NaN yaw below 1 m/s); one window per segment, so no drive aggregation (the archives carry no route
key); about half of the windows carry one repeated last row (599-row CSVs);
six archives labelled ICE by archive identity where the lab's merged platform mixes hybrids and gasoline cars; the two
Leaf archives are one car model; MIT-licensed data, public platform names only.

### 2.11 MoreTorque rlogs (`moretorque`)

What it is: 7,636 one-minute openpilot `rlog.bz2` segments of four community-fork platforms from the Hugging Face
dataset MoreTorque/rlogs (Hyundai Sonata 2020: 1 device, 3,031 segments; Mazda 3 2019: 7 devices, 2,497; Mazda CX-50:
3 devices, 1,194; Omoda E5: 1 device, 914; 75 GB), decoded with the lab's own drive builder logic reproduced from the
rlogs (every 10th `carState` sample, `carControl` / localizer / `modelV2` decimated to their qlog rates and held with the
lab's maximum ages) -> 7,190 windows (119.8 h) in 1,155 drives (runs of consecutive segment numbers; median 2 windows,
419 single-window drives), 12 cars (`DS_MORETORQUE.md`). Channels: all 20 from the lab's own sources - CAN `vEgo` /
`aEgo`, steering angle, CAN steering rate on the Sonata and the lab's derivative on the three forks whose CAN rate is
dead, driver torque with the per-drive q95 rule, pedal and steering flags, standstill, `das_enabled` = controller or
stock cruise enabled, vision lead from `modelV2` with radard's conversion (as on the lab's radar-less Hyundais and
Mazdas), yaw from the calibrated localizer frame (left-positive; correlation with the steering angle +0.87 to +0.98 on
every car); the Omoda fork reports an unsigned torque and never sets the pedal flags, so its torque and pedal groups are
absent (17 channels, the training dropout encoding). Labels: Sonata MID (4.62) / ICE, a training platform (`seen`, 24
lab drives); Mazda 3 BP SMALL (4.29) / ICE, CX-50 MID (4.67) / ICE and Omoda E5 SMALL (4.05) / BEV absent from the lab
inventory; no borderline; 3,193 SMALL / 3,997 MID windows, 874 BEV / 6,316 ICE; no LARGE class. Automation: `das_frac`
63 % (Sonata), 73 % (Mazda 3), 14 % (CX-50), 30 % (Omoda); the Sonata and CX-50 forks additionally steer with
lateral-only automation (`latActive` without `enabled`, not counted in the lab's `das_enabled`) on 28 % and 48 % of
their samples. Operating domain by platform: Sonata highway (mean 24.7 m/s), Mazda 3 21.9, CX-50 15.6, Omoda urban
(9.3 m/s).

Size (full 20-channel model primary; the variants and masks for comparison; window / drive / vehicle levels with the
vehicle-cluster bootstrap over 12 cars):

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall SMALL / MID % | chance uniform / majority % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (primary) | window | 7,190 | 43.4 [18.8, 78.9] | 41.1 [20.1, 51.0] | 81.1 / 13.4 | 33.3 / 55.6 | 60.0 / 54.3 |
|  | drive | 1,155 (501 / 654) | 42.7 [17.0, 79.6] | 37.6 [17.7, 49.5] | 85.6 / 9.8 | 33.3 / 56.6 | 70.8 / 67.3 |
|  | vehicle | 12 (8 / 4) | 66.7 [41.7, 91.7] | 40.0 [29.4, 47.8] | 100.0 / 0.0 | 33.3 / 66.7 | 63.5 / 59.7 |
|  | model | 4 (2 / 2) | 50.0 | 33.3 | 100.0 / 0.0 | 33.3 / 50.0 | - |
| V1 variant (lateral + speed) | window | 7,190 | 44.2 [20.3, 78.0] | 41.7 [21.7, 48.8] | 82.2 / 13.8 | 33.3 / 55.6 | 62.0 / 55.0 |
|  | drive | 1,155 | 40.7 [15.6, 77.5] | 35.1 [16.1, 45.9] | 83.2 / 8.1 | 33.3 / 56.6 | 70.6 / 66.4 |
|  | vehicle | 12 | 58.3 [33.3, 83.3] | 36.8 [25.0, 45.5] | 87.5 / 0.0 | 33.3 / 66.7 | 62.6 / 58.0 |
| V2 variant (longitudinal-only) | window | 7,190 | 27.9 [15.8, 46.2] | 31.7 [18.0, 37.8] | 47.6 / 12.2 | 33.3 / 55.6 | 41.7 / 34.9 |
|  | vehicle | 12 | 41.7 [16.7, 66.7] | 31.2 [15.4, 42.1] | 62.5 / 0.0 | 33.3 / 66.7 | 43.2 / 32.8 |
| V3 variant (speed-only) | window | 7,190 | 36.2 [16.0, 66.7] | 34.0 [16.9, 43.8] | 68.4 / 10.4 | 33.3 / 55.6 | 47.1 / 41.3 |
|  | drive | 1,155 | 36.8 [12.1, 72.8] | 30.9 [12.4, 42.9] | 78.6 / 4.7 | 33.3 / 56.6 | 53.4 / 45.1 |
|  | vehicle | 12 | 66.7 [41.7, 91.7] | 40.0 [29.4, 47.8] | 100.0 / 0.0 | 33.3 / 66.7 | 54.4 / 45.6 |
| full model + V1 mask (secondary) | window | 7,190 | 51.5 [28.1, 84.0] | 49.4 [27.4, 57.3] | 88.4 / 22.0 | 33.3 / 55.6 | 63.2 / 51.2 |
|  | vehicle | 12 | 66.7 [41.7, 91.7] | 40.0 [29.4, 47.8] | 100.0 / 0.0 | 33.3 / 66.7 | 60.8 / 50.8 |
| full model + V2 mask (secondary) | window | 7,190 | 44.2 [18.6, 80.4] | 39.1 [18.8, 47.5] | 85.4 / 11.3 | 33.3 / 55.6 | 53.6 / 33.2 |
| full model + V3 mask (secondary) | window | 7,190 | 43.4 [11.6, 88.1] | 33.1 [11.2, 52.7] | 93.6 / 3.4 | 33.3 / 55.6 | 55.1 / 30.1 |

Powertrain (full model primary; one BEV car, so macro-F1 and the BEV recall rest on the Omoda alone):

| model | level | units | accuracy % [95 % CI] | macro-F1 % [95 % CI] | recall BEV / ICE % | chance uniform / majority % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (primary) | window | 7,190 | 72.4 [48.8, 90.8] | 45.2 [37.0, 48.6] | 8.0 / 81.3 | 50.0 / 87.8 | 82.5 / 80.7 |
|  | drive | 1,155 (89 / 1,066) | 85.9 [66.1, 94.4] | 48.5 [42.4, 49.3] | 4.5 / 92.7 | 50.0 / 92.3 | 93.7 / 93.3 |
|  | vehicle | 12 (1 / 11) | 91.7 [75.0, 100.0] | 47.8 [40.0, 47.8] | 0.0 / 100.0 | 50.0 / 91.7 | 93.9 / 93.4 |
|  | model | 4 (1 / 3) | 75.0 | 42.9 | 0.0 / 100.0 | 50.0 / 75.0 | - |
| V1 variant | window | 7,190 | 74.6 [52.2, 99.0] | 46.5 [39.2, 53.8] | 8.7 / 83.8 | 50.0 / 87.8 | 70.0 / 67.9 |
|  | drive | 1,155 | 84.6 [64.7, 99.3] | 48.0 [41.0, 51.9] | 4.5 / 91.3 | 50.0 / 92.3 | 74.6 / 73.7 |
| V2 variant | window | 7,190 | 77.7 [58.9, 89.6] | 55.1 [51.0, 59.8] | 27.7 / 84.6 | 50.0 / 87.8 | 76.7 / 74.1 |
|  | drive | 1,155 | 87.5 [70.4, 95.4] | 55.7 [52.9, 58.5] | 18.0 / 93.3 | 50.0 / 92.3 | 90.0 / 89.0 |
| V3 variant | window | 7,190 | 73.3 [59.4, 84.3] | 54.9 [49.9, 63.0] | 38.9 / 78.1 | 50.0 / 87.8 | 77.0 / 74.6 |
|  | drive | 1,155 | 80.6 [65.1, 92.3] | 51.8 [47.2, 59.1] | 21.3 / 85.6 | 50.0 / 92.3 | 89.5 / 88.6 |
| full + V1 / V2 / V3 mask (secondary) | drive | 1,155 | 86.0 / 88.6 / 88.4 | 47.4 / 48.4 / 46.9 | BEV 2.2 / 2.2 / 0.0, ICE 93.0 / 95.8 / 95.8 | 50.0 / 92.3 | 71.2 / 67.5 / 63.3 |

Every variant and mask puts all 12 cars on the same side: the 11 ICE cars ICE, the Omoda ICE (vehicle accuracy 91.7 %
= the ICE share in every row).

Per platform (full models; mean probability over the platform's windows -> decision; drive accuracy in brackets; the
variants' decisions in the last columns):

| platform (class / pt, stratum) | cars | drives | win | size, full: P(S, M, L) % -> decision (drive acc %) | size V1 / V3 | powertrain, full: P(BEV) % -> decision (drive acc %) | P(BEV) % V1 / V2 / V3 |
|---|---|---|---|---|---|---|---|
| Hyundai Sonata (MID / ICE, seen) | 1 | 297 | 2,926 | 52, 23, 25 -> SMALL miss (13.8) | SMALL 28 / SMALL 24 | 31 -> ICE OK (90.6) | 37 / 30 / 32 |
| Mazda CX-50 (MID / ICE, absent) | 3 | 357 | 1,071 | 64, 22, 14 -> SMALL miss, 0 of 3 cars (6.4) | SMALL 23 / SMALL 27 | 11 -> ICE OK, 3 of 3 (95.5) | 8 / 18 / 30 |
| Mazda 3 (SMALL / ICE, absent) | 7 | 412 | 2,319 | 75, 18, 7 -> SMALL OK, 7 of 7 cars (91.7) | SMALL 66 (6 of 7) / SMALL 53 | 24 -> ICE OK, 7 of 7 (91.7) | 6 / 20 / 22 |
| Omoda E5 (SMALL / BEV, absent) | 1 | 89 | 874 | 55, 24, 20 -> SMALL OK (57.3) | SMALL 59 / SMALL 42 | 20 -> ICE miss (4.5) | 17 / 34 / 41 |

Per car (full models; `DAS` = `das_frac`, `lateral-only` = share of samples with the fork's lateral-only automation):

| car | model | size / pt | stratum | drives | win | DAS / lateral-only % | size decision (P S, M, L %) | size drive acc % | pt decision (P BEV %) | pt drive acc % |
|---|---|---|---|---|---|---|---|---|---|---|
| veh_1c1bc25c | Hyundai Sonata | MID / ICE | seen | 297 | 2,926 | 63 / 28 | SMALL (52, 23, 25) MISS | 13.8 | ICE (31) OK | 90.6 |
| veh_f20695f6 | Mazda CX-50 | MID / ICE | absent | 110 | 555 | 2 / 83 | SMALL (65, 26, 8) MISS | 8.2 | ICE (7) OK | 99.1 |
| veh_f1e45994 | Mazda CX-50 | MID / ICE | absent | 227 | 419 | 33 / 0 | SMALL (65, 16, 19) MISS | 5.3 | ICE (17) OK | 93.4 |
| veh_d632de27 | Mazda CX-50 | MID / ICE | absent | 20 | 97 | 1 / 70 | SMALL (56, 26, 18) MISS | 10.0 | ICE (7) OK | 100.0 |
| veh_8a7fb9d8 | Mazda 3 | SMALL / ICE | absent | 184 | 1,211 | 71 / 23 | SMALL (78, 17, 5) OK | 95.1 | ICE (26) OK | 92.9 |
| veh_95a2afec | Mazda 3 | SMALL / ICE | absent | 150 | 878 | 77 / 0 | SMALL (74, 18, 8) OK | 91.3 | ICE (18) OK | 95.3 |
| veh_5adb5f7d | Mazda 3 | SMALL / ICE | absent | 32 | 174 | 80 / 13 | SMALL (52, 24, 23) OK | 93.8 | ICE (43) OK | 75.0 |
| veh_1e5bb0ca | Mazda 3 | SMALL / ICE | absent | 28 | 35 | 5 / 0 | SMALL (63, 27, 10) OK | 82.1 | ICE (17) OK | 92.9 |
| veh_8b094748 | Mazda 3 | SMALL / ICE | absent | 8 | 11 | 0 / 100 | SMALL (63, 28, 9) OK | 62.5 | ICE (24) OK | 62.5 |
| veh_a1531e81 | Mazda 3 | SMALL / ICE | absent | 7 | 7 | 26 / 31 | SMALL (56, 29, 16) OK | 71.4 | ICE (24) OK | 85.7 |
| veh_660276e0 | Mazda 3 | SMALL / ICE | absent | 3 | 3 | 98 / 0 | SMALL (87, 5, 8) OK | 100.0 | ICE (9) OK | 100.0 |
| veh_93d1a56f | Omoda E5 | SMALL / BEV | absent | 89 | 874 | 30 / 11 | SMALL (55, 24, 20) OK | 57.3 | ICE (20) MISS | 4.5 |

Automation state and speed regime (full models, window level; `human` = das_frac < 0.5 and lateral-only < 0.5,
`lateral-only` = lateral-only >= 0.5, `engaged` = das_frac >= 0.5; the full tables are in `RESULTS_TABLES.md`):

| model | human: n, size acc %, P(MID), P(BEV) | lateral-only: n, size acc %, P(MID), P(BEV) | engaged: n, size acc %, P(MID), P(BEV) | P(MID) by mean speed 0-10 / 10-20 / 20-30 / 30+ m/s |
|---|---|---|---|---|
| Hyundai Sonata (MID / ICE) | 234, 29.1, 0.35, 0.08 | 839, 25.1, 0.28, 0.07 | 1,853, 6.5, 0.19, 0.45 | 0.37 / 0.29 / 0.20 / 0.18 |
| Mazda CX-50 (MID / ICE) | 356, 12.1, 0.22, 0.11 | 568, 16.2, 0.26, 0.07 | 147, 0.0, 0.07, 0.27 | 0.30 / 0.21 / 0.14 / 0.18 |
| Mazda 3 (SMALL / ICE) | 316, 76.3, 0.29, 0.13 | 293, 84.6, 0.23, 0.27 | 1,710, 90.4, 0.15, 0.26 | 0.30 / 0.21 / 0.12 / 0.16 |
| Omoda E5 (SMALL / BEV) | 614, 63.7, 0.26, 0.22 | 9, 88.9, 0.07, 0.33 | 251, 62.5, 0.22, 0.14 | 0.34 / 0.08 / 0.11 / - |

Sensitivity (full models, same windows): with the lead group masked the decisions do not move (Sonata P(ICE) 0.71, CX-50
0.90, Omoda P(BEV) 0.21; Mazda 3 SMALL 7 of 7); with the torque and pedal groups masked on every car (the Omoda's state)
neither do they (Omoda 0.20; Sonata P(MID) 0.26, CX-50 0.13); with the yaw group masked (yaw rate, curvature, lateral
acceleration) the size decision collapses - Sonata LARGE 0.47, CX-50 LARGE 0.48, Mazda 3 SMALL for 4 of 7 cars (0.43),
Omoda SMALL 0.41, window accuracy 24.3 % - while the powertrain calls stay (Omoda 0.22). The `--das any` re-cut that
counts the forks' lateral-only automation in `das_enabled` (`moretorque_dasany`; `das_frac` 91.2 % Sonata, 86.5 % Mazda 3,
63.2 % CX-50, 30.6 % Omoda) leaves every vehicle decision in place: Sonata P(MID) 0.23 -> 0.22, CX-50 0.22 -> 0.32 (drive
accuracy 6.4 -> 23.2 %), Mazda 3 SMALL for 6 of 7 cars, Omoda P(BEV) 0.19; window accuracy 42.4 % (size) and 71.6 %
(powertrain).

Reading:

* **SMALL is read correctly on every SMALL car** of a fleet the model never saw (Mazda 3: 7 cars, 2,319 windows, drive
  accuracy 91.7 %, P(SMALL) 0.75; Omoda E5 0.55 with the pedal and torque groups absent), by the full model and by
  every variant.
* **MID is not recognised, not even for a training platform.** The Sonata (1 car, 48.8 h; HYUNDAI_SONATA has 24 lab
  training drives) is SMALL with P(MID) 0.23, the three CX-50 (an absent model) with 0.22; no automation state (human,
  lateral-only, engaged) and no speed regime (0-10 to 30+ m/s) raises P(MID) above 0.37, and relabelling the lateral-only
  automation as engaged does not change the call. The masks show that the decision is made from the yaw group
  (removing it flips both MID cars to LARGE and halves the Mazda 3 hits), so the lateral channels are read, but on these
  forks they read as SMALL: the Sonata is driven at 24.7 m/s on highways (the lab's Sonata windows are mixed driving
  with a 91 % lead share against 67 % here), the CX-50 is steered by the fork's lateral-only automation half of the
  time. A seen platform on a new device and a new software fork is therefore not "in distribution" for the size model,
  which is the device-level warning the commaCarSegments plan (section 4) was meant to test at scale.
* **Powertrain: ICE right for 11 of 11 cars, the one BEV missed.** The ICE platforms are ICE with P(ICE) 0.69-0.89
  (0.57-0.93 per car; drive accuracy 90.6-95.5 % per platform), also under the fork's lateral-only automation and in the engaged windows of the Sonata (P(BEV)
  0.45 on its 1,853 engaged highway windows, 0.08 on its human windows - the engaged-highway drift towards BEV of 3.1 is
  visible here too, but it does not flip the car). The Omoda E5 is ICE for the full model (P(BEV) 0.20), V1 (0.17), V2
  (0.34) and V3 (0.41); its 10-20 m/s windows reach 0.26. One car, one driver, 15 h of urban driving at 9.3 m/s on a
  Chery fork whose pedal and torque flags are dead and whose vehicle parameters are placeholders: a single-car miss
  like the internal Tesla Model S (P(BEV) 0.46 with the full model), not a fleet statement.

Caveats: one community-fork fleet (openpilot forks with fork-specific flags, a dead CAN steering rate on three
platforms, an unsigned torque and dead pedal flags on the Omoda, placeholder `carParams` geometry), few devices (1 / 7 /
3 / 1) with driver and car confounded, no interval for the single-car models (the 12-car bootstrap intervals are wide);
consecutive-segment drives are short (median 2 windows), so the drive level sits close to the window level; vision lead
and calibrated-frame yaw (as on the lab's radar-less cars); the lateral-only automation of two forks is outside the lab's
`das_enabled` definition (checked by the re-cut); operating domains differ by platform (Sonata highway, Omoda urban),
so a per-platform number also reflects the roads; the Sonata is a training platform (its result is "seen model, new
device, new fork", not a held-out test); no LARGE class and one BEV; the Omoda logs are undated; Apache-2.0 data,
pseudonymous car ids only.

### 2.12 commaCarSegments, device-disjoint selection (`ccs`)

What it is: comma's commaCarSegments release (one-minute `rlog.zst` segments of the openpilot fleet, 188,883 segments
in 230 index keys), read through the owner's selection of `CCS_SELECTION.md`: legacy keys folded with the opendbc
MIGRATION table (185 platforms), the 464 devices that also appear in the lab's raw library removed (18,158 segments), then
for every model with training / validation drives in the lab (`seen`) at most 400 segments per platform from at least 20
devices in pieces of up to 5 consecutive segments of one route, and every segment for the lab's 41 held-out test models
(`held-out`) and for models outside the lab inventory (`absent`); the budget list caps six large held-out platforms at
1,000 segments. Downloaded: 49,563 segments from 4,334 devices and 23,302 routes (74 GB). Decoded by `ccs_decode.py`
(`CCS_PIPELINE.md`: opendbc 0.3.1 in its own venv, the brand's `CarInterface` / `RadarInterface` replayed on the CAN
stream, 6 workers, 3.2 h): 49,563 found -> 47,006 decoded -> 1,243 dropped by the lab's moving rule (fewer than 20 % of
the samples at v >= 1 m/s) -> **45,763 windows (762.7 h) from 4,061 devices, 21,987 drives (device x route), 113 car
models and 161 platforms**; one window = one segment, `vehicle_id` = device pseudonym. Skipped at decode time: 2,557
segments, of which 2,460 with `canValid` below 0.5 (2,457 exactly 0.00), 73 with stale 10 Hz grid samples (the lab gap
rule) and 24 whose logged fingerprint has no CarInterface in opendbc 0.3.1 (20 Tucson 4th-gen segments, while the
platform's other 345 decoded; HONDA_CRV_HYBRID_6G 3; HYUNDAI_SONATA_HEV_2024 1). The `canValid` skips are not spread
evenly: every segment of nine Honda platforms (Civic 2016 / Bosch / 2022, CR-V 5G and 2016, Fit, HR-V 2016 and 3G, Freed:
2,198 segments) failed to validate under opendbc 0.3.1 (the parser waited for a gearbox message these cars never send;
cause and recovery in 2.13), so the Honda Civic and Fit (seen), the HR-V (a held-out test model) and the Freed (absent) are
missing from this set and the CR-V is present only through its hybrid platform;
the remaining losses are partial (RAM 2500 / 3500 HD 64 of 143, Pacifica 2018 42 of 389, Octavia 31 of 322, 1-14
elsewhere). 172 selected platforms -> 161 in the set.

Channels: all 20 of the training list from the lab's own sources where the platform provides them. CAN `vEgo` / `aEgo`
(100 Hz, sampled to the 10 Hz grid), steering angle and CAN steering rate (the rate is 0 on Ford and Nissan and rarely
set on Volkswagen group and Hyundai cars, as in the lab's logs of those brands), pedal and steering-pressed flags,
standstill, `das_enabled` = stock cruise enabled or panda `controlsAllowed` (the lab definition; openpilot's own engaged
flag is not in these logs); driver torque normalised per route batch (95th percentile over moving steer-pressed samples,
at least 50 samples): present on 38,724 windows (84.6 %), absent on 7,039 (15.4 %, hands-off minutes). Yaw rate,
curvature and lateral acceleration are derived from the steering angle with the bicycle model and the platform's
`steerRatio` / wheelbase on every window (`CarState.yawRate` is filled only on Ford, 4.4 % of the windows; training used
the IMU / pose yaw), so the `mask-yaw` run is the sensitivity check for this group. Lead group: radar tracks through an
in-path gate on 15,539 windows (34.0 %: Toyota / Lexus 12,483, Honda 1,286, Ford 915, older Hyundai 794, Rivian 61),
the stock-ACC object of 33 radar-less Hyundai / Kia / Genesis platforms on 4,740 (10.4 %; the assumed sign of the
relative speed is confirmed by a median correlation of +0.70 between the gap derivative and the reported value), no lead
on 25,484 (55.7 %: the vision-only platforms Tesla, Rivian, GM, Subaru, Nissan, Mazda, the Volkswagen group, Chrysler,
radar-less Honda and E-GMP Hyundai / Kia / Genesis, where training's camera model gave a lead); the lead channels are
present on 21,550 windows (47.1 %; 1,271 windows carry the group with no object reported in that minute: 928 on stock-ACC
platforms, 343 on radar platforms). Channels per window: 20 on 18,323 windows, 19 (no torque) on 3,227, 14 (no lead) on 20,401, 13 on 3,812.
Labels: footprint and model from the lab platform table for 43,195 windows (94.4 %) and from `specs_external.csv`
(public specs, +-0.05 m^2) for 2,568 (5.6 %, 22 platforms, among them 13 of the 14 absent models; the absent Volkswagen
Touran is in the lab table); powertrain per platform as
in training (MIXED platforms unlabelled). Size labels on every window: SMALL 20,036 (50 models) / MID 15,159 (33) /
LARGE 10,568 (30); BEV vs ICE on 29,585 windows: BEV 5,420 (15 models, 621 devices) / ICE 24,165 (73 models, 1,891
devices); HEV 9,386, PHEV 2,392, FCEV 296 and MIXED / unknown 4,104 windows are scored but unlabelled for that task;
16 borderline models (5,335 windows). Strata: seen 25,667 windows / 3,199 devices / 11,627 drives / 70 models / 100
platforms; held-out 18,710 / 864 / 9,697 / 29 (of the 41 test models) / 47; absent 1,386 / 36 / 663 / 14 / 14 (10 SMALL,
the Buick Regal Sportback MID, Dodge Durango / Cadillac Escalade / Hyundai Staria LARGE; 10 ICE, no BEV; Mirai, CT 200h,
Prius v and Ioniq Plug-in unlabelled). Automation: `das_frac` mean 0.39 (seen 0.40, held-out 0.38, absent 0.39); 39.3 %
of the windows are engaged for most of the minute (18,004 against 27,759 human-driven), so this is the first external set
with both states on the same devices. Operating domain: mean speed 18.0 m/s (seen 18.2, held-out 17.8, absent 16.5;
the lab's windows 17.6), moving share 0.93, standstill 5.7 % of the samples, gas pressed on 39 %, brake on 13 %, driver
steering on 22 %. Evidence per device: median 5 windows (seen 4, held-out 15, absent 49); 513 devices with one window,
1,371 with 2-4, 822 with 5-9, 1,355 with 10 or more; 55.0 % of the drives are one window (1-5 segments per route by
construction). 91 devices appear under more than one platform (88 under two, 3 under three; 88 under more than one car
model, 60 under two size classes: a dongle
moved between cars, or a changed fingerprint); the level tables give such a device its majority label, the per-model
and pooled-decision tables count device x model pairs (4,152 for size, 2,547 for powertrain).

The full 20-channel models are the primary run (all channels present); the variants and masks are comparisons. Vehicle
= device; 95 % intervals from the vehicle-cluster bootstrap (300 resamples per stratum; the score files' 1,000-resample
intervals on the pooled set agree within 0.3 points).

(a) Full models per stratum and level (acc / macro-F1 %; "models right" per class; the internal reference is the lab's
held-out test with the same checkpoints: size 60.0 / 54.3 window, 70.8 / 67.3 drive, 63.5 / 59.7 vehicle, 28 of 41
models; powertrain 82.5 / 80.7, 93.7 / 93.3, 93.9 / 93.4, 38 of 39 models). Answers: how much of the internal number
survives on a fleet of devices the lab never saw, per stratum.

Size, full GRU S9:

| stratum | windows: acc / F1 | drives: acc / F1 | devices: acc [95 % CI] / F1 [95 % CI] | models: acc [95 % CI] / F1 [95 % CI] | models right S / M / L | device recall S / M / L % |
|---|---|---|---|---|---|---|
| all | 45,763: 49.6 / 43.2 | 21,987: 51.3 / 44.0 | 4,061: 54.1 [52.7, 55.6] / 46.4 [44.9, 48.2] | 113: 55.8 [53.0, 58.7] / 45.6 [41.2, 49.8] | 49/50, 8/33, 6/30 | 91.1 / 27.6 / 23.5 |
| seen | 25,667: 53.3 / 46.1 | 11,627: 56.6 / 47.5 | 3,199: 53.9 [52.1, 55.7] / 47.3 [45.3, 49.2] | 70: 57.1 [52.9, 60.7] / 47.1 [40.7, 51.6] | 31/32, 4/20, 5/18 | 90.8 / 28.3 / 25.8 |
| held-out | 18,710: 42.3 / 37.1 | 9,697: 42.8 / 37.8 | 864: 53.8 [50.9, 57.4] / 41.6 [37.9, 45.0] | 29: 44.8 [40.0, 51.9] / 40.0 [33.3, 49.7] | 8/8, 4/12, 1/9 | 91.1 / 25.6 / 13.2 |
| absent | 1,386: 79.7 / 36.5 | 663: 82.5 / 33.8 | 36: 88.9 [77.8, 97.2] / 31.4 [29.2, 49.3] | 14: 71.4 [63.6, 90.9] / 27.8 [25.9, 47.5] | 10/10, 0/1, 0/3 | 100.0 / 0.0 / 0.0 |

Chance 33.3; majority share 43.8 (windows, all) / 43.4 (devices) / 44.2 (models); the absent stratum is 92.0 % SMALL.
Without the 16 borderline models: 3,678 devices 57.3 [55.7, 59.3] / 49.5, 99 models 56.6 / 47.3.

Powertrain, full ResNet B5 (the absent stratum has no BEV, so no macro-F1):

| stratum | windows: acc / F1 | drives: acc / F1 | devices: acc [95 % CI] / F1 [95 % CI] | models: acc [95 % CI] / F1 [95 % CI] | models right BEV / ICE | device recall BEV / ICE % |
|---|---|---|---|---|---|---|
| all | 29,585: 81.8 / 74.7 | 14,355: 86.6 / 79.4 | 2,512: 87.8 [86.6, 89.1] / 84.7 [83.3, 86.3] | 88: 97.7 [96.2, 100.0] / 96.0 [93.4, 100.0] | 14/15, 72/73 | 86.6 / 88.2 |
| seen | 15,640: 82.2 / 75.9 | 7,031: 86.7 / 79.5 | 2,041: 86.1 [84.7, 87.6] / 83.5 [81.8, 85.1] | 51: 98.0 [95.8, 100.0] / 96.5 [93.2, 100.0] | 8/9, 42/42 | 86.4 / 86.0 |
| held-out | 13,070: 81.2 / 73.9 | 6,895: 86.4 / 79.9 | 463: 94.2 [92.0, 96.2] / 90.0 [86.2, 93.6] | 27: 100.0 [96.0, 100.0] / 100.0 [93.3, 100.0] | 6/6, 21/21 | 86.2 / 95.8 |
| absent | 875: 83.3 / - | 429: 87.2 / - | 22: 95.5 [86.4, 100.0] / - | 10: 90.0 [83.3, 100.0] / - | -, 9/10 | - / 95.5 |

Chance 50.0; majority share 81.7 (windows) / 75.3 (devices) / 83.0 (models). The two model-level misses are the Honda e
(one device, 2 windows, P(BEV) 0.34) and the absent Cadillac Escalade (one engaged highway window, P(BEV) 0.93).

(b) Variants and masks, device and model level per stratum (device acc / F1 %; models right S / M / L or BEV / ICE;
internal = the lab's held-out vehicles and models). Answers: does the full model's internal advantage over the reduced
inputs carry to the new fleet, and which channel groups carry the decision.

Size:

| run | all: devices; models | seen: devices; models | held-out: devices; models | absent: devices; models | internal: vehicles; models of 41 |
|---|---|---|---|---|---|
| full (20 ch) | 54.1 / 46.4; 49/8/6 | 53.9 / 47.3; 31/4/5 | 53.8 / 41.6; 8/4/1 | 88.9 / 31.4; 10/0/0 | 63.5 / 59.7; 28 |
| V1 lateral + speed | 48.2 / 44.2; 39/18/4 | 48.1 / 45.1; 24/10/4 | 47.8 / 38.8; 6/8/0 | 75.0 / 29.0; 9/0/0 | 62.6 / 58.0; 25 |
| V2 longitudinal-only | 42.0 / 38.9; 39/7/11 | 42.8 / 39.5; 25/5/8 | 38.0 / 33.8; 7/2/1 | 63.9 / 44.5; 7/0/2 | 43.2 / 32.8; 17 |
| V3 speed-only | 53.7 / 50.0; 46/5/14 | 55.5 / 52.7; 29/2/9 | 45.6 / 32.6; 7/2/2 | 88.9 / 75.6; 10/1/3 | 54.4 / 45.6; 20 |
| full + mask-V1 | 48.2 / 33.2; 50/1/0 | 47.5 / 33.9; 32/1/0 | 50.0 / 30.6; 8/0/0 | 88.9 / 31.4; 10/0/0 | 60.8 / 50.8; 23 |
| full + mask-V2 | 42.9 / 25.4; 50/0/0 | 42.3 / 26.0; 32/0/0 | 43.3 / 22.5; 8/0/0 | 83.3 / 30.8; 10/0/0 | 48.0 / 23.2; 15 |
| full + mask-V3 | 43.5 / 21.5; 50/1/0 | 42.3 / 21.3; 32/1/0 | 46.4 / 21.7; 8/0/0 | 88.9 / 31.8; 10/0/0 | 49.2 / 25.3; 15 |
| full + mask-yaw (sensitivity) | 39.8 / 37.5; 7/6/30 | 42.1 / 39.9; 4/5/18 | 31.5 / 28.7; 0/1/9 | 30.6 / 20.5; 3/0/3 | - |
| full + mask-lead (sensitivity) | 53.8 / 45.5; 50/7/6 | 53.7 / 46.6; 32/4/5 | 52.8 / 39.4; 8/3/1 | 88.9 / 31.4; 10/0/0 | - |

Powertrain:

| run | all: devices; models BEV / ICE | seen | held-out | absent (ICE only) | internal: vehicles; models of 39 |
|---|---|---|---|---|---|
| full (20 ch) | 87.8 / 84.7; 14/15, 72/73 | 86.1 / 83.5; 8/9, 42/42 | 94.2 / 90.0; 6/6, 21/21 | 95.5; 9/10 | 93.9 / 93.4; 38 |
| V1 lateral + speed | 76.4 / 74.0; 15/15, 49/73 | 79.0 / 77.1; 9/9, 32/42 | 66.3 / 60.8; 6/6, 11/21 | 59.1; 6/10 | 75.8 / 74.6; 33 |
| V2 longitudinal-only | 84.6 / 80.6; 14/15, 72/73 | 83.2 / 79.9; 9/9, 41/42 | 89.2 / 81.3; 5/6, 21/21 | 100.0; 10/10 | 91.4 / 90.4; 38 |
| V3 speed-only | 92.6 / 90.1; 14/15, 73/73 | 92.4 / 90.5; 9/9, 42/42 | 92.9 / 86.2; 5/6, 21/21 | 100.0; 10/10 | 91.0 / 90.1; 38 |
| full + mask-V1 | 82.4 / 71.6; 3/15, 73/73 | 81.9 / 72.8; 3/9, 42/42 | 83.6 / 58.8; 0/6, 21/21 | 100.0; 10/10 | 69.3 / 61.2; 32 |
| full + mask-V2 | 77.5 / 62.6; 0/15, 72/73 | 76.1 / 63.2; 0/9, 42/42 | 82.7 / 55.2; 0/6, 21/21 | 95.5; 9/10 | 69.3 / 56.8; 30 |
| full + mask-V3 | 81.5 / 66.6; 0/15, 72/73 | 80.6 / 67.7; 0/9, 42/42 | 84.4 / 55.7; 0/6, 21/21 | 95.5; 9/10 | 65.2 / 49.8; 31 |
| full + mask-yaw (sensitivity) | 86.2 / 83.1; 14/15, 73/73 | 84.1 / 81.5; 8/9, 42/42 | 94.6 / 91.1; 6/6, 21/21 | 100.0; 10/10 | - |
| full + mask-lead (sensitivity) | 91.3 / 88.6; 14/15, 72/73 | 90.1 / 87.6; 8/9, 42/42 | 96.3 / 93.4; 6/6, 21/21 | 95.5; 9/10 | - |

(c) The 29 held-out test models against their internal result (full models; `internal_per_model.md`: drive accuracy % /
model-level decision with P(true); external: devices right (accuracy %) and model-level decision with P(true); a model
may span several platforms, e.g. gasoline and hybrid ports, and the powertrain columns use only its labelled platforms).
Answers: does the external replica of the `split_platform` protocol agree with the lab's own held-out test, model by model.

| held-out model | size / pt label | size, ccs: devices right (acc %), decision (P) | size, internal: drive acc / decision (P) | pt, ccs: devices right (acc %), decision (P) | pt, internal |
|---|---|---|---|---|---|
| Chevrolet Equinox | SMALL / ICE | 5/5 (100.0), SMALL (0.64) OK | 50 / SMALL (0.37) | 5/5 (100.0), ICE (0.80) OK | 100 / ICE (0.84) |
| Ford F-150 Lightning | LARGE / BEV | 5/6 (83.3), LARGE (0.55) OK | 95 / LARGE (0.73) | 4/6 (66.7), BEV (0.52) OK | 86 / BEV (0.63) |
| Ford Ranger | LARGE / ICE | 2/3 (66.7), SMALL (0.38) miss | 90 / LARGE (0.74) | 3/3 (100.0), ICE (0.67) OK | 90 / ICE (0.70) |
| Genesis G80 | LARGE / ICE | 1/7 (14.3), SMALL (0.26) miss | 0 / MID (0.13) miss | 7/7 (100.0), ICE (0.73) OK | 100 / ICE (0.73) |
| Genesis G90 | LARGE / ICE | 0/9 (0.0), SMALL (0.06) miss | 0 / MID (0.13) miss | 9/9 (100.0), ICE (0.73) OK | 100 / ICE (0.74) |
| Genesis GV60 | MID / BEV | 1/2 (50.0), MID (0.52) OK | 97 / MID (0.61) | 2/2 (100.0), BEV (0.78) OK | 99 / BEV (0.83) |
| Genesis GV70 | MID / ICE | 1/8 (12.5), SMALL (0.36) miss | 67 / MID (0.45) | 8/8 (100.0), ICE (0.79) OK | 100 / ICE (0.73) |
| Genesis GV80 | LARGE / ICE | 0/3 (0.0), SMALL (0.06) miss | 62 / LARGE (0.51) | 3/3 (100.0), ICE (0.89) OK | 84 / ICE (0.73) |
| Hyundai Azera | MID / HEV | 0/1 (0.0), SMALL (0.32) miss | 80 / MID (0.50) | - (5 windows, unlabelled) | - |
| Hyundai Ioniq | SMALL / BEV (EV port) | 26/26 (100.0), SMALL (0.70) OK | 69 / SMALL (0.59) | 5/5 (100.0), BEV (0.83) OK | 90 / BEV (0.79) |
| Hyundai Kona | SMALL / BEV (EV ports; 2 of the 20 devices are the gasoline Kona 2022, ICE) | 18/20 (90.0), SMALL (0.63) OK | 86 / SMALL (0.60) | 18/20 (90.0), BEV (0.75) OK | 70 / ICE (0.67): the lab's Kona test cars are mostly gasoline, labelled ICE; not comparable |
| Hyundai Santa Cruz | LARGE / ICE | 0/1 (0.0), SMALL (0.09) miss | 83 / LARGE (0.45) | 1/1 (100.0), ICE (0.84) OK | 100 / ICE (0.69) |
| Hyundai Santa Fe | MID / ICE | 39/62 (62.9), MID (0.46) OK | 66 / MID (0.43) | 43/46 (93.5), ICE (0.73) OK | 95 / ICE (0.73) |
| Jeep Grand Cherokee | SMALL (border) / ICE | 26/36 (72.2), SMALL (0.49) OK | 22 / LARGE (0.23) miss | 36/36 (100.0), ICE (0.85) OK | 100 / ICE (0.93) |
| Kia Carnival | LARGE / ICE | 0/15 (0.0), SMALL (0.07) miss | 77 / LARGE (0.50) | 15/15 (100.0), ICE (0.80) OK | 96 / ICE (0.82) |
| Kia K5 | MID / ICE | 0/3 (0.0), SMALL (0.26) miss | 25 / SMALL (0.40) miss | 3/3 (100.0), ICE (0.77) OK | 100 / ICE (0.75) |
| Kia Optima | MID / ICE | 0/5 (0.0), SMALL (0.30) miss | 85 / MID (0.56) | 5/5 (100.0), ICE (0.70) OK | 100 / ICE (0.86) |
| Kia Sorento | MID / ICE | 0/17 (0.0), SMALL (0.26) miss | 57 / SMALL (0.40) miss | 9/9 (100.0), ICE (0.83) OK | 50 / ICE (0.66) |
| Kia Stinger | MID / ICE | 10/10 (100.0), MID (0.56) OK | 0 / SMALL (0.33) miss | 10/10 (100.0), ICE (0.70) OK | 100 / ICE (0.77) |
| Lexus RX | MID / ICE | 7/124 (5.6), SMALL (0.32) miss | 17 / SMALL (0.32) miss | 60/69 (87.0), ICE (0.69) OK | 100 / ICE (0.67) |
| Nissan Altima | MID / ICE | 0/4 (0.0), SMALL (0.32) miss | 0 / SMALL (0.20) miss | 4/4 (100.0), ICE (0.83) OK | 83 / ICE (0.63) |
| Nissan Leaf | SMALL / BEV | 17/28 (60.7), SMALL (0.54) OK | 61 / SMALL (0.56) | 22/28 (78.6), BEV (0.63) OK | 64 / BEV (0.60) |
| Tesla Model Y | MID / BEV | 17/21 (81.0), MID (0.59) OK | 66 / MID (0.45) | 20/21 (95.2), BEV (0.84) OK | 100 / BEV (0.83) |
| Toyota Avalon | MID (border) / ICE | 1/37 (2.7), SMALL (0.29) miss | 50 / SMALL (0.44) miss | 6/6 (100.0), ICE (0.82) OK | 100 / ICE (0.92) |
| Toyota Corolla | SMALL / ICE | 259/272 (95.2), SMALL (0.65) OK | 95 / SMALL (0.72) | 39/39 (100.0), ICE (0.81) OK | 94 / ICE (0.72) |
| Toyota Sienna | LARGE / ICE | 15/94 (16.0), SMALL (0.19) miss | 45 / LARGE (0.42) | 58/62 (93.5), ICE (0.75) OK | 100 / ICE (0.82) |
| Volkswagen Atlas | LARGE / ICE | 0/32 (0.0), SMALL (0.09) miss | 0 / SMALL (0.14) miss | 32/32 (100.0), ICE (0.86) OK | 100 / ICE (0.89) |
| Volkswagen Jetta | SMALL / ICE | 12/12 (100.0), SMALL (0.72) OK | 82 / SMALL (0.65) | 12/12 (100.0), ICE (0.88) OK | 100 / ICE (0.88) |
| Volkswagen Passat | SMALL (border) / MIXED | 8/8 (100.0), SMALL (0.68) OK | 81 / SMALL (0.56) | - (unlabelled) | - |

Size: 13 of 29 held-out models right here against 19 of the same 29 internally; the two tests agree on 19 models (11 hits,
8 misses), the external set loses the Ranger, GV70, GV80, Azera, Santa Cruz, Carnival, Optima and Sienna (LARGE minivans
and three-row SUVs, MID sedans) and gains the Jeep Grand Cherokee and the Kia Stinger. Powertrain: 27 of 27 here and
27 of 27 internally, each on its own label (the Kona is BEV here through its EV ports and ICE in the lab's test set: a label
difference, not a disagreement).

(d) Seen models ("seen platform, new device") and absent models (public-spec footprints), full models. Answers: does a
training platform on a device the lab never saw behave as in-distribution, and how do models outside the inventory fare.

Size, seen: 40 of 70 models right at model level (SMALL 31/32, MID 4/20, LARGE 5/18); median per-model device accuracy
59.6 %; 1,743 of 3,245 device x model pairs (53.7 %). The ten seen models with the lowest P(true):

| seen model | label | devices | windows | devices right (acc %) | P(true) | decision |
|---|---|---|---|---|---|---|
| Volkswagen Transporter | LARGE (border) | 2 | 74 | 0/2 (0.0) | 0.07 | SMALL miss |
| Cadillac Escalade ESV | LARGE | 1 | 8 | 0/1 (0.0) | 0.08 | SMALL miss |
| Honda Odyssey | LARGE | 96 | 381 | 2/96 (2.1) | 0.08 | SMALL miss |
| Honda Pilot | LARGE (border) | 77 | 396 | 2/77 (2.6) | 0.09 | SMALL miss |
| Honda Ridgeline | LARGE | 56 | 392 | 2/56 (3.6) | 0.17 | SMALL miss |
| Ford Explorer | LARGE | 44 | 374 | 5/44 (11.4) | 0.21 | SMALL miss |
| Chrysler Pacifica | LARGE | 184 | 1,475 | 27/184 (14.7) | 0.21 | SMALL miss |
| Cadillac XT4 | MID (border) | 1 | 1 | 0/1 (0.0) | 0.21 | SMALL miss |
| Ford Maverick | LARGE (border) | 32 | 382 | 2/32 (6.2) | 0.21 | SMALL miss |
| Subaru Ascent | MID | 34 | 389 | 0/34 (0.0) | 0.21 | SMALL miss |

The seen MID hits are the Tesla Model 3 (17/28 devices, P(MID) 0.53), Hyundai Ioniq 5 (46/101, 0.49), Kia EV6 (70/87,
0.63) and Ford Mustang Mach-E (7/12, 0.46; borderline); the seen LARGE hits the Chevrolet Silverado 1500 (31/52, 0.42),
RAM 1500 (57/99, 0.47), RAM 2500 / 3500 HD (4/5, 0.49), Ford F-150 (25/42, 0.44) and Chevrolet Traverse (1/1, 0.46). Seen
ICE MID sedans and crossovers are SMALL: Accord 55/120 devices MID, Highlander 42/225, Camry 12/167, Lexus ES 21/68,
Sonata 4/56, Tucson 3/27, Acura RDX 0/21, Ascent 0/34, Superb 0/10, G70 0/4; Ioniq 6 (MID BEV) 7/24.

Powertrain, seen: 50 of 51 models (BEV 8/9: the Honda e misses; ICE 42/42); median per-model device accuracy 96.4 %; 1,772
of 2,059 device x model pairs (86.1 %). Lowest P(true) after the Honda e (0.34): Ford Bronco Sport (ICE, 7/8 devices,
0.62), Toyota Highlander (ICE, 148/225 devices, 0.64), Cadillac Escalade ESV (0.67), Toyota RAV4 (ICE, 224/307, 0.68),
Ford Focus (0.69), Volkswagen Transporter (0.72), Lexus IS (14/16, 0.72), Nissan Rogue (0.73), Chevrolet Bolt EUV (BEV,
118/160 devices, 0.73; 380 windows over 160 devices, median 2 per device).

Absent models (public-spec footprints; every one):

| absent model | label | devices | windows | size: devices right, decision (P true) | pt: devices right, decision (P true) |
|---|---|---|---|---|---|
| Skoda Octavia | SMALL / ICE | 7 | 286 | 7/7, SMALL (0.77) OK | 7/7, ICE (0.81) OK |
| Toyota Mirai | SMALL / FCEV | 6 | 296 | 6/6, SMALL (0.66) OK | unlabelled: 36.5 % of windows BEV, 0/6 devices |
| SEAT Ateca | SMALL / ICE | 4 | 152 | 4/4, SMALL (0.75) OK | 4/4, ICE (0.82) OK |
| Volkswagen T-Roc | SMALL / ICE | 3 | 147 | 3/3, SMALL (0.78) OK | 3/3, ICE (0.84) OK |
| Hyundai Elantra GT / i30 | SMALL / ICE | 2 | 100 | 2/2, SMALL (0.69) OK | 2/2, ICE (0.58) OK |
| Hyundai Ioniq Plug-in | SMALL / PHEV | 2 | 100 | 2/2, SMALL (0.70) OK | unlabelled: 39.0 % BEV, 1/2 devices |
| Lexus CT 200h | SMALL / HEV | 3 | 65 | 3/3, SMALL (0.55) OK | unlabelled: 30.8 % BEV, 1/3 |
| Toyota Prius v | SMALL / HEV | 3 | 50 | 3/3, SMALL (0.63) OK | unlabelled: 60.0 % BEV, 2/3 |
| Skoda Kamiq | SMALL / ICE | 1 | 35 | 1/1, SMALL (0.86) OK | 1/1, ICE (0.77) OK |
| Volkswagen Touran | SMALL (border) / ICE | 1 | 44 | 1/1, SMALL (0.70) OK | 1/1, ICE (0.77) OK |
| Buick Regal Sportback | MID / ICE | 1 | 50 | 0/1, SMALL (0.32) miss | 1/1, ICE (0.78) OK |
| Hyundai Staria | LARGE / ICE | 1 | 49 | 0/1, SMALL (0.04) miss | 1/1, ICE (0.68) OK |
| Dodge Durango | LARGE / ICE | 1 | 11 | 0/1, SMALL (0.22) miss | 1/1, ICE (0.84) OK |
| Cadillac Escalade | LARGE / ICE | 1 | 1 | 0/1, SMALL (0.19) miss | 0/1, BEV (0.07) miss |

(e) Lead source and automation state per stratum (full models; `human` = `das_frac` < 0.5, `engaged` >= 0.5; device
accuracy inside a cell uses the device's windows of that cell). Answers: does the lead construction or the automation
state carry the decision.

| factor | value | stratum | size: windows / devices | size: window acc % | size: device acc % (right / n) | pt: windows / devices | pt: window acc % | pt: device acc % (right / n) |
|---|---|---|---|---|---|---|---|---|
| lead | radar | all | 15,539 / 1,760 | 43.4 | 47.6 (837/1,760) | 8,078 / 900 | 77.9 | 82.3 (741/900) |
|  |  | seen | 8,044 / 1,237 | 47.8 | 44.6 (552/1,237) | 4,712 / 721 | 76.7 | 79.6 (574/721) |
|  |  | held-out | 7,085 / 531 | 36.5 | 53.7 (285/531) | 3,366 / 185 | 79.7 | 92.4 (171/185) |
| lead | stock ACC | all | 4,740 / 267 | 54.0 | 53.6 (143/267) | 3,352 / 207 | 73.8 | 86.5 (179/207) |
|  |  | seen | 1,666 / 150 | 53.7 | 46.0 (69/150) | 1,075 / 126 | 74.0 | 85.7 (108/126) |
|  |  | held-out | 2,963 / 113 | 52.8 | 61.9 (70/113) | 2,194 / 79 | 74.3 | 88.6 (70/79) |
| lead | none | all | 25,484 / 2,526 | 52.6 | 57.9 (1,462/2,526) | 18,155 / 1,700 | 85.0 | 89.6 (1,524/1,700) |
|  |  | seen | 15,957 / 2,103 | 56.1 | 59.2 (1,245/2,103) | 9,853 / 1,367 | 85.8 | 88.7 (1,212/1,367) |
|  |  | held-out | 8,662 / 409 | 43.5 | 49.1 (201/409) | 7,510 / 317 | 84.0 | 93.1 (295/317) |
| state | human | all | 27,759 / 3,486 | 51.0 | 55.7 (1,943/3,486) | 18,356 / 2,150 | 92.2 | 93.8 (2,017/2,150) |
|  |  | seen | 15,361 / 2,694 | 55.3 | 56.1 (1,510/2,694) | 9,673 / 1,697 | 92.4 | 92.6 (1,572/1,697) |
|  |  | held-out | 11,562 / 785 | 43.3 | 53.9 (423/785) | 8,187 / 444 | 91.9 | 97.5 (433/444) |
| state | engaged | all | 18,004 / 2,700 | 47.6 | 48.3 (1,303/2,700) | 11,229 / 1,620 | 64.8 | 70.9 (1,149/1,620) |
|  |  | seen | 10,306 / 2,019 | 50.5 | 49.2 (993/2,019) | 5,967 / 1,235 | 65.7 | 70.9 (876/1,235) |
|  |  | held-out | 7,148 / 675 | 40.9 | 43.7 (295/675) | 4,883 / 374 | 63.3 | 70.6 (264/374) |

The absent rows (radar 12 devices, stock ACC 4, none 25; human 34, engaged 30) are in `RESULTS_TABLES.md`. Same split for
the other powertrain runs (window accuracy human / engaged; device accuracy human / engaged, computed from the
`probs__*__ens__*.npz` files): V3 86.0 / 73.5 and 94.1 / 84.3; V2 78.3 / 68.8 and 88.7 / 73.9; full + mask-lead 93.2 / 73.3
and 94.8 / 81.8; mean P(BEV) on ICE windows: full 0.12 human / 0.39 engaged, V3 0.17 / 0.31, mask-lead 0.10 / 0.32.

(f) Evidence per device (full models; devices right / devices, accuracy %, by labelled windows per device). Answers:
how many minutes of a new device the decision needs, and whether more minutes help.

| task | stratum | devices | 1 window | 2-4 windows | 5-9 windows | 10+ windows |
|---|---|---|---|---|---|---|
| size | all | 4,061 | 308/513 (60.0) | 771/1,371 (56.2) | 438/822 (53.3) | 681/1,355 (50.3) |
| size | seen | 3,199 | 250/424 (59.0) | 650/1,225 (53.1) | 372/736 (50.5) | 452/814 (55.5) |
| size | held-out | 864 | 59/93 (63.4) | 136/170 (80.0) | 70/90 (77.8) | 200/511 (39.1) |
| size | absent | 36 | 2/3 (66.7) | 2/2 (100.0) | - | 28/31 (90.3) |
| powertrain | all | 2,512 | 277/346 (80.1) | 681/830 (82.0) | 433/491 (88.2) | 814/845 (96.3) |
| powertrain | seen | 2,041 | 245/308 (79.5) | 650/796 (81.7) | 417/475 (87.8) | 446/462 (96.5) |
| powertrain | held-out | 463 | 34/40 (85.0) | 35/40 (87.5) | 19/20 (95.0) | 348/363 (95.9) |
| powertrain | absent | 22 | 0/1 (0.0) | 1/1 (100.0) | - | 20/20 (100.0) |

(g) Predicted-share confusion for size (windows; predicted class shares per true class and mean probabilities), full
model and full model with the yaw group masked. Answers: does the model use the whole label space, and how much of the
decision rests on the steering-derived yaw / curvature / lateral-acceleration group.

| run | stratum | true class | devices | windows | predicted share SMALL, MID, LARGE % | mean P(SMALL, MID, LARGE) |
|---|---|---|---|---|---|---|
| full | all | SMALL | 1,781 | 20,036 | 78.1, 19.9, 1.9 | 0.66, 0.29, 0.05 |
| full | all | MID | 1,291 | 15,159 | 65.1, 32.6, 2.3 | 0.56, 0.38, 0.06 |
| full | all | LARGE | 1,049 | 10,568 | 54.1, 25.8, 20.1 | 0.47, 0.31, 0.22 |
| full | seen | SMALL | 1,347 | 12,141 | 81.3, 17.4, 1.3 | 0.68, 0.28, 0.04 |
| full | seen | MID | 1,002 | 7,254 | 66.7, 31.2, 2.1 | 0.58, 0.37, 0.06 |
| full | seen | LARGE | 879 | 6,272 | 50.7, 24.5, 24.8 | 0.45, 0.29, 0.26 |
| full | held-out | SMALL | 406 | 6,620 | 71.0, 25.8, 3.2 | 0.61, 0.33, 0.06 |
| full | held-out | MID | 294 | 7,855 | 63.5, 33.9, 2.6 | 0.55, 0.39, 0.06 |
| full | held-out | LARGE | 168 | 4,235 | 58.9, 27.8, 13.2 | 0.51, 0.33, 0.16 |
| full + mask-yaw | all | SMALL | 1,781 | 20,036 | 27.1, 19.0, 53.9 | 0.28, 0.25, 0.47 |
| full + mask-yaw | all | MID | 1,291 | 15,159 | 20.4, 22.8, 56.8 | 0.24, 0.27, 0.48 |
| full + mask-yaw | all | LARGE | 1,049 | 10,568 | 15.7, 14.2, 70.1 | 0.20, 0.21, 0.59 |
| full + mask-yaw | seen | SMALL | 1,347 | 12,141 | 27.8, 17.7, 54.5 | 0.29, 0.24, 0.47 |
| full + mask-yaw | seen | MID | 1,002 | 7,254 | 21.8, 25.0, 53.1 | 0.25, 0.28, 0.46 |
| full + mask-yaw | seen | LARGE | 879 | 6,272 | 13.2, 12.8, 74.0 | 0.18, 0.20, 0.62 |
| full + mask-yaw | held-out | SMALL | 406 | 6,620 | 23.9, 21.9, 54.1 | 0.26, 0.26, 0.47 |
| full + mask-yaw | held-out | MID | 294 | 7,855 | 19.1, 20.8, 60.1 | 0.23, 0.27, 0.50 |
| full + mask-yaw | held-out | LARGE | 168 | 4,235 | 19.4, 16.3, 64.4 | 0.23, 0.24, 0.54 |

(The absent rows: SMALL 85.2 / 13.8 / 1.0, MID (1 device) 68.0 / 32.0 / 0.0, LARGE (3 devices) 75.4 / 19.7 / 4.9.)

(h) Unlabelled powertrains, platforms with at least 100 windows (full model; share of the platform's windows called BEV,
devices whose mean probability is BEV, mean P(BEV)). Answers: what the BEV-vs-ICE model reads hybrids, plug-ins, the
fuel-cell Mirai and the merged gasoline / electrified platforms as. For scale: 17.5 % of the labelled ICE windows and 78.7 %
of the labelled BEV windows are called BEV.

| platform | model | powertrain | stratum | devices | windows | windows BEV % | devices BEV | mean P(BEV) |
|---|---|---|---|---|---|---|---|---|
| TOYOTA_RAV4_PRIME | Toyota RAV4 | PHEV | seen | 40 | 384 | 59.1 | 28/40 | 0.56 |
| HYUNDAI_SANTA_FE_PHEV_2022 | Hyundai Santa Fe | PHEV | held-out | 8 | 301 | 52.2 | 5/8 | 0.51 |
| HYUNDAI_IONIQ_PHEV | Hyundai Ioniq | PHEV | held-out | 9 | 292 | 51.0 | 4/9 | 0.50 |
| CHEVROLET_VOLT | Chevrolet Volt | PHEV | seen | 24 | 386 | 40.2 | 6/24 | 0.42 |
| HYUNDAI_IONIQ_PHEV_2019 | Hyundai Ioniq Plug-in | PHEV | absent | 2 | 100 | 39.0 | 1/2 | 0.42 |
| CHRYSLER_PACIFICA_2019_HYBRID | Chrysler Pacifica | PHEV | seen | 96 | 388 | 38.7 | 21/96 | 0.40 |
| CHRYSLER_PACIFICA_2018_HYBRID | Chrysler Pacifica | PHEV | seen | 29 | 378 | 33.3 | 3/29 | 0.36 |
| HYUNDAI_SONATA_HYBRID | Hyundai Sonata | HEV | seen | 13 | 328 | 65.9 | 10/13 | 0.62 |
| HYUNDAI_ELANTRA_HEV_2021 | Hyundai Elantra | HEV | seen | 11 | 334 | 56.6 | 6/11 | 0.54 |
| HYUNDAI_IONIQ_HEV_2022 | Hyundai Ioniq | HEV | held-out | 8 | 304 | 51.0 | 3/8 | 0.50 |
| TOYOTA_PRIUS_TSS2 | Toyota Prius | HEV | seen | 99 | 399 | 46.4 | 39/99 | 0.47 |
| HYUNDAI_SANTA_FE_HEV_2022 | Hyundai Santa Fe | HEV | held-out | 8 | 262 | 45.0 | 3/8 | 0.47 |
| KIA_NIRO_HEV_2021 | Kia Niro | HEV | seen | 4 | 144 | 42.4 | 1/4 | 0.46 |
| KIA_NIRO_HEV_2ND_GEN | Kia Niro | HEV | seen | 8 | 187 | 38.5 | 0/8 | 0.42 |
| LEXUS_ES_TSS2 | Lexus ES | HEV | seen | 62 | 375 | 37.3 | 25/62 | 0.40 |
| TOYOTA_CAMRY_TSS2 | Toyota Camry | HEV | seen | 81 | 391 | 37.1 | 24/81 | 0.39 |
| TOYOTA_PRIUS | Toyota Prius | HEV | seen | 109 | 368 | 37.0 | 32/109 | 0.41 |
| TOYOTA_AVALON_2019 | Toyota Avalon | HEV | held-out | 25 | 774 | 34.4 | 6/25 | 0.38 |
| TOYOTA_SIENNA_4TH_GEN | Toyota Sienna | HEV | held-out | 32 | 879 | 33.6 | 2/32 | 0.36 |
| TOYOTA_RAV4H | Toyota RAV4 | HEV | seen | 37 | 389 | 33.2 | 5/37 | 0.35 |
| TOYOTA_COROLLA_TSS2 | Toyota Corolla | HEV | held-out | 233 | 971 | 31.9 | 60/233 | 0.34 |
| HYUNDAI_IONIQ | Hyundai Ioniq | HEV | held-out | 4 | 104 | 26.9 | 0/4 | 0.30 |
| TOYOTA_RAV4_TSS2_2022 | Toyota RAV4 | HEV | seen | 67 | 384 | 25.0 | 12/67 | 0.32 |
| LEXUS_RX | Lexus RX | HEV | held-out | 55 | 968 | 20.0 | 4/55 | 0.23 |
| KIA_SORENTO_HEV_4TH_GEN | Kia Sorento | HEV | held-out | 8 | 287 | 16.7 | 0/8 | 0.27 |
| TOYOTA_CAMRY | Toyota Camry | HEV | seen | 86 | 390 | 14.6 | 7/86 | 0.21 |
| HONDA_CRV_HYBRID | Honda CR-V | HEV | seen | 13 | 304 | 13.2 | 1/13 | 0.26 |
| HONDA_INSIGHT | Honda Insight | HEV | seen | 15 | 334 | 9.9 | 1/15 | 0.19 |
| HONDA_ACCORD | Honda Accord | HEV | seen | 120 | 390 | 9.7 | 4/120 | 0.19 |
| TOYOTA_MIRAI | Toyota Mirai | FCEV | absent | 6 | 296 | 36.5 | 0/6 | 0.41 |
| FORD_ESCAPE_MK4 | Ford Escape | MIXED | seen | 13 | 269 | 36.8 | 5/13 | 0.39 |
| FORD_MAVERICK_MK1 | Ford Maverick | MIXED | seen | 32 | 382 | 29.3 | 4/32 | 0.32 |
| HYUNDAI_TUCSON_4TH_GEN | Hyundai Tucson | MIXED | seen | 27 | 321 | 24.9 | 5/27 | 0.31 |
| KIA_SPORTAGE_5TH_GEN | Kia Sportage | MIXED | seen | 9 | 277 | 23.5 | 1/9 | 0.28 |
| FORD_EXPLORER_MK6 | Ford Explorer | MIXED | seen | 44 | 374 | 23.3 | 3/44 | 0.28 |
| VOLKSWAGEN_CRAFTER_MK2 | Volkswagen Crafter | MIXED | seen | 4 | 115 | 22.6 | 1/4 | 0.25 |
| TOYOTA_ALPHARD_TSS2 | Toyota Alphard | MIXED | seen | 6 | 165 | 20.6 | 0/6 | 0.25 |
| TOYOTA_AVALON_TSS2 | Toyota Avalon | unknown | held-out | 6 | 237 | 19.8 | 0/6 | 0.24 |
| VOLKSWAGEN_ARTEON_MK1 | Volkswagen Arteon | MIXED | seen | 4 | 134 | 18.7 | 0/4 | 0.28 |
| TOYOTA_CHR | Toyota C-HR | MIXED | seen | 7 | 156 | 17.3 | 0/7 | 0.24 |
| VOLKSWAGEN_PASSAT_MK8 | Volkswagen Passat | MIXED | held-out | 8 | 256 | 16.4 | 2/8 | 0.21 |
| FORD_F_150_MK14 | Ford F-150 | MIXED | seen | 42 | 387 | 15.0 | 1/42 | 0.21 |
| VOLKSWAGEN_GOLF_MK7 | Volkswagen Golf | MIXED | seen | 40 | 391 | 12.3 | 0/40 | 0.20 |
| VOLKSWAGEN_TIGUAN_MK2 | Volkswagen Tiguan | MIXED | seen | 31 | 385 | 11.4 | 1/31 | 0.17 |
| AUDI_A3_MK3 | Audi A3 | MIXED | seen | 6 | 237 | 7.6 | 0/6 | 0.14 |

Pooled over all unlabelled platforms (53): PHEV 44.6 % of 2,392 windows called BEV (69 of 214 devices), HEV 32.0 % of
9,386 (245 of 1,093), FCEV 36.5 % of 296 (0 of 6), MIXED 19.9 % of 3,867 (23 of 275).

Reading:

* **Powertrain at the scale of a fleet: the model-level result of the lab's held-out test is reproduced, the device-level
  number is six points lower.** Full model: 86 of 88 car models (97.7 %; BEV 14/15, ICE 72/73) against 38 of 39
  internally; devices 87.8 % [86.6, 89.1] / macro-F1 84.7 against 93.9 / 93.4 internal vehicle accuracy; windows 81.8 /
  74.7 against 82.5 / 80.7. The held-out stratum, the external replica of the `split_platform` protocol, gives 27 of 27
  models and 94.2 % [92.0, 96.2] of 463 devices (macro-F1 90.0), i.e. the internal vehicle-level number (93.9) on devices
  and software the lab never saw. The seen stratum is lower (86.1 % of 2,041 devices, 50 of 51 models): a training
  platform on a new device is not easier than a new platform; the seen devices are dominated by Toyota ICE platforms with
  many single-window devices (RAV4 TSS2 65 of 113 devices right, Highlander 148 of 225, Bolt EUV 118 of 160 with a
  median of 2 windows per device) where the per-device evidence is one or two minutes.
* **Engaged minutes are where the powertrain model loses, on a CAN fleet and at scale.** Window accuracy 92.2 % on the
  27,759 human-driven windows against 64.8 % on the 18,004 engaged ones; mean P(BEV) on ICE windows 0.12 human / 0.39
  engaged; the same devices split by state: 93.8 % right from their human minutes, 70.9 % from their engaged minutes.
  Internally the full model loses 0.7 points of vehicle accuracy on the engaged subset (93.2 against 93.9). The drift
  towards BEV under automation that the RAV4 sample (2.9), the Sonata's engaged windows (2.11) and the phase-1 ACC cars
  showed is therefore not a GNSS artefact; it is how the full model reads engaged minutes of a new fleet with a CAN
  `aEgo`. Part of it is the lead group as built here: with the lead masked the engaged-minute device accuracy rises
  from 70.9 to 81.8 % and the overall device accuracy from 87.8 to 91.3 % (ICE recall 88.2 -> 93.3, BEV 86.6 -> 85.3,
  model level unchanged at 86 of 88); the speed-only V3, which has no lead, is at 84.3 % on engaged minutes and 92.6 %
  overall. The lead here is a radar in-path gate or the stock-ACC object, in training it was radard's fused `leadOne`
  (and the camera model's lead on vision-only cars), so the lead features of an engaged minute are systematically
  different from training. A probe, not a decomposition: the human-minute accuracy of the lead-masked model (93.2 %)
  is also slightly above the full model's (92.2 %).
* **The pedal, torque and lead groups do not add to the powertrain decision on this fleet; the speed-only variant is the
  best model.** Devices: V3 92.6 % [91.5, 93.6] / 90.1 (internal 91.0 / 90.1), V2 84.6 / 80.6 (internal 91.4 / 90.4), full
  87.8 / 84.7 (internal 93.9 / 93.4); models: V3 87 of 88 (ICE 73/73), V2 and full 86 of 88. The full model's internal
  advantage over V3 (+2.9 vehicle points) turns into a 4.8-point deficit, V3 itself is within 1.6 points of its internal
  number, and the gap between the two is the lead group (full + mask-lead 91.3 against V3 92.6). The pedal flags, the
  reason the set was downloaded (section 4), do not show up as a gain: V2 (pedals present, no lateral channels) is below
  V3 at device level in every stratum. The masked full models stay uninformative (3.4: mask-V2 / V3 call 0 of 15 BEV
  models). The lateral + speed variant V1 over-calls BEV here (BEV 15/15 models, ICE 49/73; held-out ICE 11/21) where it
  called every commaSteeringControl archive ICE (2.10): the lateral channels make the powertrain decision fleet-dependent
  in both directions, so they stay out of a powertrain model that is meant to transfer (3.5 point 3).
* **Model-by-model the held-out BEVs behave as internally, with one repair.** Nissan Leaf: full model BEV for 22 of 28
  devices (P 0.63) while V3 and V2 call the NISSAN_LEAF platform ICE (as the internal V3 / V2 miss our own Leaf, 18-21 %
  of drives, and as VED's Leafs failed in phase 1); the commaSteeringControl Leaf archives were V3-right on engaged
  highway minutes only. F-150 Lightning 4 of 6 devices (P 0.52; internally 86 % of drives). Kona: 18 of 20 devices right,
  the two EV ports BEV for 16 of 18 devices (P 0.75) and the two gasoline Kona 2022 devices ICE; the internal Kona row is ICE because the lab's test Konas are mostly gasoline cars -
  a label difference, not a disagreement. Hyundai Ioniq (EV port) 5/5, Tesla Model Y 20/21, Genesis GV60 2/2.
* **Size: SMALL transfers, MID and LARGE do not, and the seen stratum is no better than the held-out one.** Full model:
  devices 54.1 % [52.7, 55.6] / 46.4 (internal 63.5 / 59.7), models 63 of 113 (55.8 %; internal 28 of 41 = 68.3 %):
  SMALL 49/50 models (1,606 of 1,762 devices, 91.1 %), MID 8/33 (351 of 1,274, 27.6 %), LARGE 6/30 (241 of 1,025,
  23.5 %); internally MID 6/14 and LARGE 8/12 models, 47.3 / 50.6 % vehicle recall. Seen 53.9 % of devices and 40 of 70
  models (MID 4/20, LARGE 5/18), held-out 53.8 % and 13 of 29 (MID 4/12, LARGE 1/9): both ten points below the internal
  held-out test, and the seen platforms on new devices are not read as in-distribution, which generalises the MoreTorque
  Sonata (2.11) to 70 training models and 3,199 devices. The misses are systematic per platform, not noise: Odyssey 2 of
  96 devices, Pilot 2/77, Ridgeline 2/56, Sienna 15/94, Carnival 0/15, Atlas 0/32, Lexus RX 7/124, Highlander 42/225,
  Camry 12/167, Ascent 0/34, Acura RDX 0/21 (P(true) 0.06-0.21 at model level), so more minutes per device do not help:
  size device accuracy falls from 60.0 % (one window) to 50.3 % (10 or more windows) while powertrain accuracy rises from
  80.1 to 96.3 % (table f); the held-out 10+ bucket is at 39.1 % because its many-window devices are Lexus RX, Sienna
  and Avalon cars.
* **LARGE survives for pickups only; the minivans and three-row SUVs that the internal test recognised are lost.** Hits:
  Silverado 1500 (31/52 devices, P(LARGE) 0.42), RAM 1500 (57/99, 0.47), RAM HD (4/5), F-150 (25/42, 0.44), F-150
  Lightning (5/6, 0.55) and the single Traverse device; every minivan and SUV is SMALL (Pacifica 27/184 devices, Palisade
  24/63, Explorer 5/44, Maverick 2/32, CX-9 4/13, G80 1/7, G90 0/9, GV80 0/3, Santa Cruz 0/1, Transporter 0/2). Internally
  the full model had the Carnival at 77 % and the Sienna at 45 % of drives, the Yukon at 100 %; here the Carnival is 0 of
  15 devices and the Sienna 16.0 %, so the internal minivan hits were not robust to new devices, and the "truck-like
  LARGE" reading of 3.2 (point 3) is the one that holds. MID: 6 of the 8 hits are BEV platforms (Model 3 17/28 devices,
  Model Y 17/21, Ioniq 5 46/101, EV6 70/87, GV60 1/2, Mach-E 7/12), the other two the Santa Fe (39/62) and the Stinger
  (10/10); every ICE MID sedan or crossover apart from those two is SMALL (Accord 55/120, Highlander 42/225, Lexus RX
  7/124, Camry 12/167, Sonata 4/56, Sorento 0/17, Avalon 1/37, Optima 0/5, K5 0/3, Altima 0/4, Ascent 0/34, RDX 0/21), and
  the MID BEV Ioniq 6 is a miss (7/24). Whether the MID hits on the Tesla, E-GMP and Mach-E platforms reflect size or the
  platform signature that the powertrain model also reads cannot be separated here; descriptive only.
* **The lateral channels are used, and they are what does not transfer.** With the yaw group masked the size decision
  collapses to LARGE (device recall SMALL 25.7 / MID 20.4 / LARGE 88.3 %, 30 of 30 LARGE models), a state the training
  never produced (3.4), so the mask shows the decision runs through the steering-derived group without measuring the
  derived-yaw error; the powertrain model is indifferent to it (86.2 against 87.8 % of devices, 87 of 88 models). The
  trained variants are the cleaner comparison: the speed-only V3 reaches 53.7 % [52.2, 55.3] of devices / macro-F1 50.0,
  its internal vehicle number (54.4 / 45.6), gets 14 of 30 LARGE models (the pickups and the Traverse plus the Pacifica, Rivian,
  Ranger, Durango, Staria, Crafter and both Escalades; 44.5 % LARGE device recall) and beats the full model's macro-F1 (46.4) at
  device and model level (50.2 against 45.6); the lateral + speed V1 gets 18 of 33 MID models (54.6 % MID device recall
  against the full model's 27.6) but only 4 of 30 LARGE and 39 of 50 SMALL. The `aEgo`-texture size signal of 3.3
  therefore transfers intact on a CAN fleet, as on commaSteeringControl, while the lateral size signal that lifts the full
  model nine vehicle points above V3 internally does not reach new devices through a steering-derived yaw; whether the
  cause is the derived group itself (bicycle model, no slip or roll, lab training on IMU / pose yaw) or a device- or
  driver-specific lateral signature cannot be told apart in this set (a measured yaw exists only on Ford).
* **The lead group matters for powertrain, not for size; the lead source is confounded with the brand.** Masking the lead
  leaves size unchanged (53.8 against 54.1 % of devices, 63 models right in both) and improves powertrain (above). By lead
  source, device accuracy is lowest on radar platforms (size 47.6 %, powertrain 82.3 %) and highest without a lead (57.9 /
  89.6 %), but the radar rows are 80 % Toyota / Lexus windows, whose ICE platforms are the weak ones (Highlander, RAV4,
  Lexus RX), and inside the held-out stratum the three sources are within 5 points for powertrain (92.4 / 88.6 / 93.1 %);
  the lead source is a brand label here, not a cause.
* **Hybrids are read in between, plug-ins closer to BEV.** 32.0 % of the HEV windows and 44.6 % of the PHEV windows are
  called BEV (245 of 1,093 and 69 of 214 devices), against 17.5 % of the labelled ICE and 78.7 % of the labelled BEV
  windows; the fuel-cell Mirai 36.5 % (0 of 6 devices); the merged gasoline / electrified platforms 19.9 %, i.e. on the
  ICE side. The RAV4 Prime (59.1 %, 28 of 40 devices), Sonata Hybrid (65.9 %, 10/13) and Elantra Hybrid (56.6 %) sit on
  the BEV side, the Honda hybrids (Accord 9.7 %, Insight 9.9 %, CR-V 13.2 %) and the Camry 2018 (14.6 %) on the ICE side:
  the model reads electrified drivetrains with regenerative braking as partly BEV, as the phase-1 mild hybrids suggested
  (3.1), with a spread by brand and by plug-in capacity that is consistent with the amount of regeneration but is not
  tested here.
* **Absent models (14, public-spec footprints for 13 of them): the SMALL ones are all right, the three LARGE and the one MID are not,
  and these are single-device probes.** Size 10 of 10 SMALL models (32 of 32 devices), Buick Regal MID -> SMALL (0.32),
  Staria / Durango / Escalade LARGE -> SMALL (0.04-0.22; 49, 11 and 1 windows); powertrain 9 of 10 ICE models, the
  Escalade's single engaged window BEV (0.93). The speed-only V3 calls all 14 right (Regal MID, the three LARGE), again
  from the `aEgo` texture.

Caveats: yaw, curvature and lateral acceleration are steering-derived on every window (bicycle model with the
platform's `steerRatio` / wheelbase, no slip or roll; training used the IMU / pose yaw), the one group whose transfer
cannot be separated from a model weakness here (the post-hoc understeer-corrected re-decode of 2.15 separates them: with
the yaw group divided by 1 + k v^2 the same windows give 63.2 % of the devices and 72 of 113 models, the LARGE minivans and
SUVs included, so the as-run size numbers of this section carry a known input bias); the lead is a radar in-path gate without radard's vision fusion, the
stock-ACC object on radar-less Hyundai / Kia / Genesis (sign assumed and confirmed by correlation only) and absent on
vision-only platforms (55.7 % of the windows) where training had the camera lead, and the engaged-minute results show
that this group is read differently from training; torque is normalised per route batch of 1-5 minutes (absent on 15.4 %
of the windows), not per drive; `das_enabled` is stock cruise or `controlsAllowed`, not openpilot's engaged flag; one
window per segment and 1-5 segments per route, so 55.0 % of the drives are one minute and the drive level sits close to
the window level; the per-device evidence is uneven (median 5 windows, 513 single-window devices, and the seen stratum
is capped at 400 segments per platform while the held-out one keeps up to 1,000); 91 devices appear under more than one platform
(majority label in the level tables); footprints of 13 of the 14 absent models and 9 further platforms come from public specs
(+-0.05 m^2); several platforms merge gasoline and electrified variants upstream (MIXED, unlabelled) and the lab's ICE
label of the Highlander, RAV4 TSS2 and Lexus RX platforms covers hybrid-majority mixes in some years, which may be part
of their low ICE device accuracy (65.8 / 73.0 / 87.0 %); route overlap with the lab's library is excluded by the device
filter but the openpilot software versions differ from the lab's logs (2025 ports, community branches); the Honda Civic,
Fit, HR-V and Freed are missing because their segments did not validate under opendbc 0.3.1 (2,198 segments; recovered in
2.13), and 73
segments with stale grid samples and 24 without a CarInterface were skipped (section 2.12 first paragraph, `CCS_PIPELINE.md`);
intervals are vehicle-cluster bootstraps of 300 resamples per stratum; MIT-licensed data, pseudonymous device ids and
public platform names only.

### 2.13 commaCarSegments, the nine recovered Honda platforms (`ccs_honda`)

What it is: the 2,198 budget-list segments of the nine Honda platforms whose every segment the phase-3 decode skipped with
`canValid` 0.00 (2.12): HONDA_CIVIC, HONDA_CIVIC_BOSCH, HONDA_CIVIC_2022, HONDA_CRV_5G, HONDA_FIT, HONDA_CRV, HONDA_HRV,
HONDA_HRV_3G and HONDA_FREED. Cause: opendbc 0.3.1's Honda `CarState` reads the gearbox message named by
`CarParams.transmissionType`; the logging openpilot versions left that field at `unknown` for Honda, so the parser registered
`GEARBOX_AUTO` 0x1A3 while these traditional-CVT cars broadcast `GEARBOX_CVT` 0x191 only, and the pure-Python parser holds
`canValid` at 0 for the whole segment while a registered message has never arrived. Fix: `ccs_decode.py
--honda-gearbox-from-can` (Honda only, off by default) applies 0.3.1's own `_get_params` rule to the pt-bus addresses of
the log before the interfaces are built (0x191 present and not an Acura RDX -> `cvt`, else `automatic`, neither ->
`manual`) and records the result per segment; the default path is byte-identical to the phase-3 run (same sha1 of the
window tensor and meta on 20 check segments). The 2024-08 lab openpilot car stack was used only as a cross-check
(`ccs_honda_check.py`, `CCS_PIPELINE.md` "Honda re-decode"): on 20 option-off / option-on pairs every channel is identical,
and on 21 probe segments against the legacy stack the steering, pedal, torque and derived-yaw signals are bit-identical
after the 0.5-s start-up transient and `vEgo` agrees to a 99th percentile below 1e-4 m/s, except for a `wheelSpeedFactor`
convention (the legacy stack's `vEgo` is 2.5 % higher on the CR-V 5G, CR-V 2016 and HR-V; `ccs` already carries the 0.3.1
convention, so the re-decode keeps it). The option was preferred because it keeps one parser for every set and recovers the
103 segments the legacy stack cannot decode (below). Accounting: 2,198 segments found in 931 routes of 230 devices -> 2,189
decoded (9 skipped: 5 with `canValid` 0.00, one Fit and four CR-V 5G segments of 3 devices on which the wheel-speed and
VSA messages never appear on the pt bus although the gearbox message does, a wiring matter of those devices rather than
the gearbox rule; 4 with 17-31 stale grid samples) -> 57 dropped by the moving rule (moving fraction 0-0.175) -> **2,132
windows (35.5 h) from 230 devices, 913 drives (device x route), 5 car models, 9 platforms**; 467 s with 6 workers. Gearbox
rule: `cvt` on 2,086 segments, `automatic` on 103 (every one of them HONDA_CIVIC_2022, 9 devices: they broadcast 0x1A3 and
0x184 and no 0x191, most likely the Civic Hybrid that opendbc files under the same platform name; they keep the lab's ICE
label of the platform and are reported with and without below, not relabelled; one of the nine also carries three `cvt`
windows each under HONDA_CIVIC and HONDA_CRV_5G). Labels and strata: lab table for eight platforms, public specs for the Freed;
every platform SMALL (3.74-4.28 m^2, no borderline), every labelled platform ICE (2,084 windows), the Freed unlabelled (48);
seen 1,932 windows / 224 devices / 3 models (Civic 123 devices over three platforms, CR-V 96 over two, Fit 7; two devices are counted under both the Civic and the
CR-V), held-out
152 / 5 / 1 (the HR-V, two platforms), absent 48 / 1 / 1 (the Freed). Lead: Nidec radar through the in-path gate on the
Civic 2016, Fit, HR-V 2016, Freed and CR-V 2016 (909 windows, 42.6 %, of which 887, 41.6 %, hold an in-path object in the
minute; `lead_valid` 0.39-0.77 per platform over its decoded segments), none on the four radarless Bosch ports (Civic Bosch /
2022, CR-V 5G, HR-V 3G: 1,223 windows, 57.4 %); the route-batch torque scale is absent on 406
windows (19.0 %), among them every HONDA_CRV window because that port never sets `steer_pressed`; yaw, curvature and
lateral acceleration steering-derived on every window, as in `ccs`. Automation and domain: `das_frac` mean 0.42 (20-56 %
per platform), 902 windows engaged for most of the minute against 1,230 human-driven; mean speed 18.2 m/s, gas pressed on
37 %, brake on 15 %, driver steering on 18 % of the samples of the window set. Evidence: median 6 windows per device (17 devices with one
window, 79 with 2-4, 61 with 5-9, 73 with 10 or more); 46.1 % of the drives are one window. 9 of the 230 devices also
appear in `ccs` under other platforms (same pseudonym: a dongle moved between cars), 2 appear under more than one platform inside
this set (one of them, a possible hybrid, under three); the per-platform tables count a device once per platform, the level tables once.

Same runs as `ccs` (the full 20-channel models primary, V1 / V2 / V3, the three preset masks, the yaw and lead masks);
vehicle = device; the strata intervals are vehicle-cluster bootstraps of 300 resamples (the score files' 1,000-resample
intervals on the pooled set agree within one point). One class per task, so there is no macro-F1 and the majority share is
100 %: the numbers are the SMALL and ICE recall of each run on a new Honda fleet, not a discrimination result, and the
masked full models, which call almost everything SMALL and ICE (3.4), score as high as the full model here for that reason.

(a) Decisions per platform (full models; mean probability over the platform's labelled windows -> decision; the V1 / V3 size
and V3 / V2 powertrain decisions in the last columns). Answers: whether SMALL and ICE hold on each of the nine recovered
platforms, and what the reduced-input variants make of them.

| platform (model, footprint) | stratum | lead | veh | drives | win | DAS % | size, full: P(S, M, L) % -> decision, devices right, drive acc % | size V1 / V3 | powertrain, full: P(BEV) % -> decision, devices right | pt V3 / V2 |
|---|---|---|---|---|---|---|---|---|---|---|
| HONDA_FIT (Honda Fit, 3.74) | seen | radar 99 % | 7 | 116 | 273 | 35.0 | 60, 36, 4 -> SMALL OK, 6/7, 73.3 | MID miss / SMALL | 17 -> ICE OK, 7/7 | ICE / ICE |
| HONDA_CIVIC (Honda Civic, 4.20) | seen | radar 98 % | 33 | 186 | 385 | 45.7 | 66, 30, 4 -> SMALL OK, 32/33, 83.3 | MID miss / SMALL | 18 -> ICE OK, 33/33 | ICE / ICE |
| HONDA_CIVIC_BOSCH (Honda Civic, 4.20) | seen | none | 55 | 154 | 395 | 39.7 | 75, 21, 4 -> SMALL OK, 55/55, 96.1 | SMALL / SMALL | 11 -> ICE OK, 55/55 | ICE / ICE |
| HONDA_CIVIC_2022 (Honda Civic, 4.21; 9 of the 36 devices `automatic`) | seen | none | 36 | 148 | 393 | 45.6 | 67, 28, 4 -> SMALL OK, 35/36, 87.8 | SMALL / SMALL | 21 -> ICE OK, 34/36 | ICE / ICE |
| HONDA_CRV (Honda CR-V, 4.23; torque absent) | seen | radar 89 % | 4 | 46 | 106 | 43.6 | 79, 19, 3 -> SMALL OK, 4/4, 93.5 | SMALL / SMALL | 25 -> ICE OK, 4/4 | ICE / ICE |
| HONDA_CRV_5G (Honda CR-V, 4.28) | seen | none | 92 | 154 | 380 | 45.4 | 75, 20, 4 -> SMALL OK, 91/92, 93.5 | MID miss / SMALL | 15 -> ICE OK, 91/92 | ICE / ICE |
| HONDA_HRV (Honda HR-V, 4.01) | held-out | radar 99 % | 2 | 62 | 97 | 20.4 | 74, 23, 3 -> SMALL OK, 2/2, 96.8 | MID miss / SMALL | 16 -> ICE OK, 2/2 | ICE / ICE |
| HONDA_HRV_3G (Honda HR-V, 4.17) | held-out | none | 3 | 17 | 55 | 56.2 | 73, 25, 3 -> SMALL OK, 3/3, 94.1 | MID miss / SMALL | 17 -> ICE OK, 3/3 | ICE / ICE |
| HONDA_FREED (Honda Freed, 4.06, public spec; powertrain unlabelled) | absent | radar 100 % | 1 | 30 | 48 | 25.6 | 55, 34, 10 -> SMALL OK, 1/1, 80.0 | SMALL / SMALL | 20 -> ICE (unlabelled) | ICE / ICE |

(b) Every run at device and model level per stratum (device accuracy % [95 % CI] with devices right / devices; models right;
window accuracy in the last column; internal = the lab's held-out vehicles). Answers: how far each input list carries
SMALL and ICE on this fleet, and which channel groups the full model's decision rests on.

Size (all SMALL):

| run | all: devices [CI]; models of 5 | seen: devices; models of 3 | held-out (HR-V): devices; model | absent (Freed) | window acc % | internal vehicle acc / F1 |
|---|---|---|---|---|---|---|
| full (20 ch) | 98.3 [96.5, 99.6] (226/230); 5 | 98.2 [96.4, 99.8] (220/224); 3 | 100.0 (5/5); 1 | 1/1; 1 | 83.8 | 63.5 / 59.7 |
| V1 lateral + speed | 47.8 [41.5, 54.8] (110/230); 3 | 48.2 (108/224); 2 | 20.0 (1/5); 0 | 1/1; 1 | 49.6 | 62.6 / 58.0 |
| V2 longitudinal-only | 73.0 [68.3, 79.1] (168/230); 3 | 74.1 (166/224); 2 | 40.0 (2/5); 1 | 0/1; 0 | 51.4 | 43.2 / 32.8 |
| V3 speed-only | 86.5 [82.2, 90.9] (199/230); 5 | 86.2 (193/224); 3 | 100.0 (5/5); 1 | 1/1; 1 | 64.2 | 54.4 / 45.6 |
| full + mask-V1 (secondary) | 98.7 [97.2, 99.8] (227/230); 5 | 98.7 (221/224); 3 | 100.0 (5/5); 1 | 1/1; 1 | 92.5 | 60.8 / 50.8 |
| full + mask-V2 (secondary) | 96.5 [93.9, 98.9] (222/230); 5 | 96.4 (216/224); 3 | 100.0 (5/5); 1 | 1/1; 1 | 81.7 | 48.0 / 23.2 |
| full + mask-V3 (secondary) | 97.0 [94.3, 98.7] (223/230); 5 | 96.9 (217/224); 3 | 100.0 (5/5); 1 | 1/1; 1 | 89.5 | 49.2 / 25.3 |
| full + mask-yaw (sensitivity) | 14.3 [9.8, 19.4] (33/230); 0 | 14.7 (33/224); 0 | 0.0 (0/5); 0 | 0/1; 0 | 20.2 | - |
| full + mask-lead (sensitivity) | 98.7 [97.0, 100.0] (227/230); 5 | 98.7 (221/224); 3 | 100.0 (5/5); 1 | 1/1; 1 | 86.2 | - |

Powertrain (all ICE; the Freed is unlabelled, so the absent stratum is empty):

| run | all: devices [CI]; models of 4 | seen: devices; models of 3 | held-out (HR-V): devices; model | window acc % | internal vehicle acc / F1 |
|---|---|---|---|---|---|
| full (20 ch) | 99.1 [97.8, 100.0] (227/229); 4 | 99.1 [97.8, 100.0] (222/224); 3 | 100.0 (5/5); 1 | 89.3 | 93.9 / 93.4 |
| V1 lateral + speed | 86.0 [81.2, 89.7] (197/229); 4 | 86.6 (194/224); 3 | 60.0 (3/5); 1 | 72.7 | 75.8 / 74.6 |
| V2 longitudinal-only | 97.8 [95.6, 99.6] (224/229); 4 | 97.8 (219/224); 3 | 100.0 (5/5); 1 | 83.6 | 91.4 / 90.4 |
| V3 speed-only | 98.3 [96.1, 99.6] (225/229); 4 | 98.2 (220/224); 3 | 100.0 (5/5); 1 | 88.8 | 91.0 / 90.1 |
| full + mask-V1 (secondary) | 98.3 [96.1, 99.6] (225/229); 4 | 98.2 (220/224); 3 | 100.0 (5/5); 1 | 89.9 | 69.3 / 61.2 |
| full + mask-V2 (secondary) | 99.6 [98.7, 100.0] (228/229); 4 | 99.6 (223/224); 3 | 100.0 (5/5); 1 | 86.3 | 69.3 / 56.8 |
| full + mask-V3 (secondary) | 100.0 (229/229); 4 | 100.0 (224/224); 3 | 100.0 (5/5); 1 | 94.0 | 65.2 / 49.8 |
| full + mask-yaw (sensitivity) | 98.7 [97.4, 99.6] (226/229); 4 | 98.7 (221/224); 3 | 100.0 (5/5); 1 | 87.5 | - |
| full + mask-lead (sensitivity) | 99.1 [97.8, 100.0] (227/229); 4 | 99.1 (222/224); 3 | 100.0 (5/5); 1 | 91.7 | - |

(c) The held-out HR-V against its internal result (full models; `internal_per_model.md`). Answers: whether the one
held-out test model of this set agrees with the lab's own held-out test.

| held-out model (platforms) | label | size, ccs_honda: devices right, decision (P) | size, internal: drive acc / decision (P) | pt, ccs_honda | pt, internal |
|---|---|---|---|---|---|
| Honda HR-V (HONDA_HRV, HONDA_HRV_3G) | SMALL / ICE | 5/5 (100.0), SMALL (0.73) OK; drive acc 96.2 % | 46.0 / SMALL (0.50) | 5/5 (100.0), ICE (0.84) OK; drive acc 94.9 % | 95.0 / ICE (0.84) |

(d) Lead source and automation state (full models; `human` = `das_frac` < 0.5, `engaged` >= 0.5; device accuracy inside a
cell from the device's windows of that cell). Answers: whether the radar gate or the engaged minutes move the decisions.

| factor | value | windows / devices | size: window acc % | size: device acc % (right / n) | pt: windows / devices | pt: window acc % | pt: device acc % (right / n) |
|---|---|---|---|---|---|---|---|
| lead | radar | 887 / 47 | 80.4 | 95.7 (45/47) | 839 / 46 | 87.4 | 100.0 (46/46) |
| lead | none | 1,245 / 193 | 86.2 | 97.9 (189/193) | 1,245 / 193 | 90.5 | 97.9 (189/193) |
| state | human | 1,230 / 204 | 91.1 | 98.5 (201/204) | 1,194 / 203 | 95.6 | 99.5 (202/203) |
| state | engaged | 902 / 171 | 73.7 | 84.8 (145/171) | 890 / 170 | 80.7 | 94.1 (160/170) |

Mean P(BEV) on the ICE windows 0.08 human / 0.29 engaged; mean P(SMALL) 0.73 / 0.65. The strata rows are in
`RESULTS_TABLES.md` (held-out HR-V: 100.0 % of its human windows' devices on both tasks, 4 of 5 from the engaged ones).

(e) Evidence per device (full models; devices right / devices by labelled windows per device). Answers: how many minutes of
a new Honda device the decision needs.

| task | devices | 1 window | 2-4 windows | 5-9 windows | 10+ windows |
|---|---|---|---|---|---|
| size | 230 | 15/17 (88.2) | 78/79 (98.7) | 60/61 (98.4) | 73/73 (100.0) |
| powertrain | 229 | 16/17 (94.1) | 79/79 (100.0) | 60/61 (98.4) | 72/72 (100.0) |

Reading:

* **SMALL and ICE at scale on a fleet the phase-3 decode had lost.** Full model: SMALL for all nine platforms, 226 of 230
  devices (98.3 % [96.5, 99.6]) and 5 of 5 models; ICE for all eight labelled platforms, 227 of 229 devices (99.1 %
  [97.8, 100.0]) and 4 of 4 models; the unlabelled Freed is ICE with P(BEV) 0.20. The held-out HR-V is 5 of 5 on both tasks
  with P(SMALL) 0.73 and P(ICE) 0.84, where the internal test had the same model at 46.0 % drive accuracy / SMALL 0.50 and
  95.0 % / ICE 0.84. The Honda Civic, Fit and HR-V that 2.12 listed as missing are therefore right at model level, and the
  CR-V, present in 2.12 through its hybrid port only, gets its gasoline ports (95 of 96 devices). Against `ccs` (SMALL
  device recall 91.1 %, ICE 88.2 %) these are the easy end of the fleet: small CVT cars, 58 % of the minutes human-driven,
  a radar or no lead; the window level (83.8 / 89.3 %) is where the per-minute uncertainty shows, and ten minutes of a
  device settle both tasks (table e: 73 of 73 and 72 of 72 devices with 10 or more windows).
* **The CIVIC_2022 misses are the possible hybrids.** HONDA_CIVIC_2022 is ICE for 34 of 36 devices, and both misses are
  among the 9 `automatic`-gearbox devices (P(BEV) 0.62 over 12 windows and 0.56 over 6); those nine have a mean P(BEV) of
  0.38 against 0.14 for the 27 `cvt` devices of the same platform and 72.5 % window accuracy against 90.2 % for the rest of
  the set, 7 of 9 right at platform level and 8 of 9 at device level (one of the two carries six `cvt` windows under two
  other platforms and is ICE by a hair, P(BEV) 0.485, once they are pooled). Without the nine devices the set is 219 of 220
  (99.5 %), HONDA_CIVIC_2022 27 of 27 and its model-level P(ICE) 0.84 instead of 0.79; the one remaining device-level miss
  is a CR-V 5G device with a single fully engaged window (P(BEV) 0.67). Consistent with, not proof of, the hybrid reading
  of 2.12 (h); the label stays as the lab has it.
* **The yaw mask collapses the size call, the lead mask changes nothing.** With the steering-derived yaw group masked the
  size decision falls to 33 of 230 devices (14.3 %, 0 of 5 models; 70.7 % of the windows called LARGE, mean P(LARGE) 0.58)
  while powertrain is unmoved (98.7 %); with the lead masked both tasks stay where they are (98.7 / 99.1 % of devices), as
  expected on a set that is 58 % radarless. As in 2.12 the size decision runs through the lateral group, and on an
  all-SMALL set the mask reads as the collapse to LARGE that training never produced (3.4), not as a measurement of the
  derived-yaw error (caveats).
* **Engaged minutes score lower, here on small ICE cars with a radar gate or no lead.** Window accuracy 73.7 % against
  91.1 % (size) and 80.7 % against 95.6 % (powertrain) on the 902 engaged and 1,230 human-driven windows; mean P(BEV) on
  the ICE windows 0.29 against 0.08; the same devices judged from their engaged minutes alone are 84.8 % (size) and 94.1 %
  (powertrain) right against 98.5 / 99.5 % from their human minutes. The drift of 2.12 at a smaller scale; it flips no
  model and, with ten minutes per device, hardly any device.
* **The reduced-input variants fall off for size and barely for powertrain.** Devices: V1 47.8 / 86.0 %, V2 73.0 / 97.8 %,
  V3 86.5 / 98.3 % (size / powertrain). The lateral + speed V1 calls five of the nine platforms MID (Fit, Civic 2016, CR-V
  5G, both HR-V ports; 1 of 5 held-out devices) where the full model has them SMALL with P 0.60-0.75: the lateral channels
  alone read these small cars as MID at the window level (49.6 %), the full model with the longitudinal groups does not.
  The speed-only V3 has all nine platforms SMALL and 199 of 230 devices, and 225 of 229 ICE.
* **Phase-3 pooled totals.** With `ccs_honda` the phase-3 scope (device x car-model pairs, 3.1 / 3.2) moves from
  2,232 / 2,547 to 2,460 / 2,778 for powertrain (87.6 -> 88.6 %; ICE 1,688 / 1,917 -> 1,916 / 2,148, 88.1 -> 89.2 %) and from
  2,245 / 4,152 to 2,473 / 4,384 for size (54.1 -> 56.4 %; SMALL 1,635 / 1,798 -> 1,863 / 2,030, 90.9 -> 91.8 %); MID and
  LARGE are untouched. The addition is 228 of 231 and 228 of 232 pairs, all SMALL ICE, so it raises the totals without
  changing what 2.12 says about MID and LARGE.

Caveats: one brand, one size class and one powertrain class, so no macro-F1 and no discrimination result (the majority
baseline is 100 %); the masked full models score 97-100 % because they call everything SMALL / ICE (3.4), not because the
masked channels are dispensable. Two artefacts shared with `ccs` and `comma2k19`, stated once here: sample 0 of every
window is the all-zero `CarState` of the fresh 0.3.1 parser (`v`, `a`, steering, torque and yaw are 0 at k = 0 in every window and `standstill` reads 1 there, a 1-of-600 artefact the lab's
own logs do not have), and the lead gate admits occasional oncoming traffic (relative
speed below -30 m/s on 1.2 % of the valid lead samples, lead speed below -2 m/s on 2.2 %, minimum -63 m/s; 537 windows carry
at least one such sample). The derived yaw carries the understeer bias of 2.14: the bicycle-model yaw overestimates the
lab's IMU / pose yaw (and 2018 openpilot's vehicle-model yaw on the comma2k19 logs) by 1.5-2x at highway speed, so the lateral
channels of this set are too large on fast windows relative to what the models were trained on; the understeer-corrected
re-decode `ccs_honda_us` of 2.15 moves 12 of the 230 devices out of SMALL (10 to MID, 2 to LARGE) and one MID device back
(98.3 -> 93.5 %), every platform staying SMALL,
and leaves the powertrain call in place (99.1 -> 99.6 %).
Five segments stay lost (`canValid` 0.00: wheel-speed and VSA messages absent from the pt bus on 3 devices); HONDA_CRV
carries no torque channel (4 devices, 106 windows, 19 channels); the cross-check's `wheelSpeedFactor` convention means the
CR-V 5G, CR-V 2016 and HR-V windows (583) carry a `vEgo` 2.5 % below what the 2024-08 lab stack would log, which the
training logs of those platforms may share or not; the 9 possible-hybrid devices keep the platform's ICE label; the HR-V
rests on 5 devices and the Freed on 1; 9 devices are shared with `ccs` (same pseudonym, different platform); intervals are
vehicle-cluster bootstraps of 300 resamples per stratum; MIT-licensed data, pseudonymous device ids and public platform
names only.

### 2.14 comma2k19, two cars on 2018 hardware (`comma2k19`)

What it is: comma.ai's 2018 release (MIT licence), the two chunks the owner downloaded: Chunk 1 is one device on a 2017
Toyota RAV4 with Toyota Safety Sense P (platform TOYOTA_RAV4, a seen SMALL ICE model, 188 segments in 21 route folders of
2-15 segments), Chunk 3 one device on a 2016 Honda Civic Touring (HONDA_CIVIC, seen SMALL ICE, 202 segments in 21 folders
of 1-21). The raw one-minute 2018 rlogs (CAN buses 0 and 1 plus the panda's transmit echoes, `carState` / `controlsState`
at 100 Hz, radar at 20 Hz, no camera bus) were decoded by the same decoder as `ccs` with four options that are off by
default and leave the `ccs` / `ccs_honda` path byte-identical (`CCS_PIPELINE.md` "comma2k19"): `--layout comma2k19`
(the chunk folder layout, `.bz2` logs); `--fingerprint-map` (none of the 390 segments carries a `carParams` event, because
2018 openpilot logged it in a route's first segment only and the chunks hold middle segments, so the `CarParams` of every
segment is built the way openpilot builds it at start-up, with opendbc 0.3.1's `CarInterface.get_params` from the mapped
platform and the segment's own live fingerprint: uniform provenance, and the Civic's gearbox comes out as `cvt` by 0.3.1's
own rule on all 200 segments); `--das-from controlsState` (`das_enabled` = `controlsState.enabled` or `cruiseState.enabled`,
the lab definition, with `controlsState` logged at 100 Hz; the 2018 panda state is 1 Hz and not read); and
`--ignore-absent-messages` (2018 logs have no camera bus and the openpilot-sent `PRE_COLLISION` and HUD frames sit in the
transmit echoes, never among the received frames, so 0.3.1's registered-but-never-seen HUD messages would hold `canValid`
at 0; the waiver covers only messages absent from the segment's live fingerprint, which feed no training channel, and the
checksum, counter and timeout checks of every message the log carries stay in force). Accounting: 390 segments found in 42
route folders (route sizes 1-21 segments; 8 of the 42 folders have gaps in the segment numbering, which the drive builder
flags and which cost no window) -> 387 decoded (3 skipped with 77 / 110 / 381 stale grid samples: two Civic segments of
52.0 / 48.7 s of CAN and one RAV4 segment of 21.5 s) -> 6 dropped by the moving rule (4 Civic, 2 RAV4) -> **381 windows
(6.4 h; 185 RAV4 / 196 Civic) from 2 devices and 42 drives**; 153 s with 6 workers. `can_valid` mean 0.987 / 0.997 (RAV4 /
Civic); radar lead on both cars (TSS-P and Nidec radar on bus 1; `lead_valid` 0.78 / 0.87); `das_frac` 0.56 / 0.62 with
`cruise_frac` and `controls_enabled_frac` equal to two decimals on both cars (0.55 / 0.55, 0.61 / 0.61; 2018 openpilot engaged
with the stock cruise button), 105 / 121
windows engaged against 80 / 75 human-driven; mean speed 22.8 / 20.9 m/s; all 20 channels on every window, the torque scale
never absent. Checks against the logged 2018 `carState` of the same segments (five per car, nearest-time match, sample 0
excluded): `vEgo` within 0.22 m/s at the maximum and 0.11 m/s at the 99th percentile with a median decoder / logged ratio
of 1.0000 on every segment (both equal the mean wheel speed; on ten further segments checked by the review with `comma2k19_yaw_check.py` the maxima
reach 0.5 m/s on single samples, the 99th percentiles stay at or below 0.08 m/s, the median ratio is 0.9994-1.0002, and the
steering angle agrees to 0.6 deg at the 99th percentile with correlation 0.998-1.000 and offset 0.0); the 2018 `aEgo` is noisier (standard deviation 1.1-1.6x the
decoder's) and lags the decoder's by 0.2-0.3 s, so the two correlate 0.64-0.94 only (the speed-filter gains changed between
2018 openpilot and opendbc 0.3.1; the decoder's `aEgo` is the modern filter the training logs carry); steering angle
correlation 0.9996-1.0000 with a median offset of 0.00 deg; the Civic's steering torque has the **opposite sign** in the
2018 log (opendbc later inverted the Honda torque-sensor sign so that positive turns left as on the other brands; the
decoder applies today's DBC to the 2018 bytes and so carries the training convention, and `steer_pressed`, from the
absolute value, agrees); gas / brake / `steer_pressed` / standstill differ on at most 2.2 % of the samples (median 0.2 %, at
the pedal transitions); `das_enabled` against the log's own flags differs on at most 0.7 %. The set is scored as a scope of
its own ("old software"), outside the phase-3 pooled scope and inside "all sets".

Results (full model primary; window / drive / device levels with the two cars' probabilities; a two-car bootstrap is
uninformative, so the drive-level interval is the only one quoted):

| task | run | window acc % | drive acc % [95 % CI] | devices | Toyota RAV4 2017: decision (P), drive acc % | Honda Civic 2016: decision (P), drive acc % |
|---|---|---|---|---|---|---|
| powertrain | full (20 ch) | 70.9 | 88.1 [85.7, 90.5] | 2/2 ICE | ICE (0.68), 85.7 | ICE (0.71), 90.5 |
| powertrain | V3 speed-only | 64.3 | 83.3 [76.2, 90.5] | 2/2 | ICE (0.62), 90.5 | ICE (0.64), 76.2 |
| powertrain | V1 / V2 | 60.4 / 69.6 | 81.0 / 83.3 | 2/2 / 2/2 | ICE / ICE | ICE / ICE |
| powertrain | full + mask-V1 / V2 / V3 (secondary) | 96.6 / 85.6 / 94.5 | 100.0 / 95.2 / 100.0 | 2/2 each | ICE | ICE |
| powertrain | full + mask-yaw / mask-lead | 78.0 / 92.7 | 95.2 / 100.0 | 2/2 each | ICE | ICE |
| powertrain | full, human-driven windows (155) | 100.0 (mean P(BEV) 0.06) | - | - | P(BEV) 0.06 (80 windows) | 0.05 (75) |
| powertrain | full, engaged windows (226) | 50.9 (mean P(BEV) 0.47) | - | - | P(BEV) 0.51 (105) | 0.44 (121) |
| size | full (20 ch) | 51.7 | 54.8 [52.4, 57.1] | 1/2 | SMALL (0.54, 0.44, 0.02), 57.1 | MID (0.485, 0.491, 0.02) miss, 52.4 |
| size | V3 speed-only | 82.2 | 97.6 [95.2, 100.0] | 2/2 | SMALL (0.53, 0.23, 0.24) | SMALL (0.54, 0.24, 0.23) |
| size | V1 / V2 | 44.4 / 47.5 | 28.6 / 57.1 | 0/2 / 2/2 | MID / SMALL | MID / SMALL |
| size | full + mask-V1 / V2 / V3 (secondary) | 97.6 / 87.7 / 96.6 | 100.0 / 97.6 / 100.0 | 2/2 each | SMALL | SMALL |
| size | full + mask-yaw / mask-lead | 3.4 / 54.9 | 0.0 / 59.5 | 0/2 (LARGE) / 2/2 | LARGE / SMALL | LARGE / SMALL |
| size | full, human-driven windows (155) | 89.0 | - | - | P(SMALL) 0.72, P(MID) 0.26 | 0.68, 0.29 |
| size | full, engaged windows (226) | 26.1 | - | - | P(SMALL) 0.40, P(MID) 0.58 | 0.37, 0.61 |

Derived yaw against the logged one (the 2018 logs carry `carState.yawRate`; `UNDERSTEER.md` section 9 shows that this field is
the output of 2018 openpilot's single-track vehicle model, reproduced from the same steering angle and speed to a relative
error of 3e-8, not an inertial sensor; `ccs` carries a `yawRate` on Ford only): ratio of the decoder's bicycle-model yaw rate
to the logged yaw rate by speed bin, both cars; correlation 0.999, same sign everywhere.

| speed bin (m/s) | 1-5 | 10-15 | 20-25 | 25-30 | 30-40 |
|---|---|---|---|---|---|
| Toyota RAV4 2017 | 0.98 | 1.13 | 1.54 | 1.85 | 2.06 |
| Honda Civic 2016 | 0.87 | 1.03 | 1.53 | 1.78 | 2.01 |

The review re-derived this table over all 381 windows with `comma2k19_yaw_check.py` (lab `LogReader`; the decoder's grid matched
to the logged `carState` by nearest time, sample 0 excluded): the per-bin medians of the sample ratios are within 0.03 of the
values above, the least-squares slopes 1.00 / 1.13 / 1.54 / 1.82 / 2.08 (RAV4) and 0.88 / 1.02 / 1.47 / 1.76 / 2.07 (Civic), k
0.0009-0.0010 s^2/m^2, the within-bin correlation at least 0.94 with the same sign on every sample above 0.02 rad/s; the bicycle
model recomputed from the logged steering angle and speed gives the same ratios, so the bias is the kinematic model's, not the
decoder's sampling (`yaw_check.json` on 5 segments per car and `yaw_check_all.json` in the work folder).

Reading:

* **ICE for both cars, with the engaged minutes under a 2018 longitudinal controller at a coin flip.** The full model calls
  the Civic ICE with P 0.71 (90.5 % of its 21 drives) and the RAV4 with 0.68 (85.7 %), and every variant and mask agrees at
  car level; but the 155 human-driven windows are 100.0 % right with mean P(BEV) 0.06 while the 226 engaged windows are
  50.9 % right at 0.47. In these logs openpilot did the longitudinal control itself (the transmit echoes carry
  `BRAKE_COMMAND` on the Civic and `ACC_CONTROL` on the RAV4 in every probed segment, although today's `CarParams` files the
  TSS-P RAV4 under stock longitudinal), so the engaged minutes are a 2018 controller the models never saw: the
  engaged-minute drift of 2.12 and 2.13 at its strongest, on two cars. The speed-only V3 is not better here (64.3 % of the
  windows, 83.3 % of the drives; P(ICE) 0.62 / 0.64), so on these two cars the drift is not only the lead group.
* **Size: the Civic is a coin flip with the full model, both cars SMALL with the speed-only variant.** Civic P(SMALL) 0.485
  against P(MID) 0.491 -> MID; RAV4 SMALL 0.54. The split by state says where the MID call comes from: the human-driven
  windows of both cars are SMALL with P 0.68-0.72 (89.0 % window accuracy), the engaged ones MID with 0.58-0.61 (26.1 %),
  i.e. the engaged highway minutes of a small car read as MID to the full model, as the MID sedans of commaSteeringControl
  read as SMALL (2.10). The speed-only V3 has both cars SMALL (0.53 / 0.54; 97.6 % of the drives) from the acceleration
  texture; the full model under the V3 mask is SMALL too, as it is for everything (3.4); the yaw mask turns both cars LARGE
  (3.4 % of the windows SMALL), the pattern of 2.12 and 2.13.
* **The main lesson of the set is the derived yaw.** The bicycle-model yaw matches the logged `yawRate` at low speed
  (ratio 0.98 / 0.87 at 1-5 m/s) and overestimates it with speed: 1.13 / 1.03 at 10-15 m/s, 1.54 / 1.53 at 20-25, 1.85 /
  1.78 at 25-30, 2.06 / 2.01 at 30-40 (RAV4 / Civic; correlation 0.999, same sign). The ratio grows as 1 + k v^2 with k of
  about 0.001 s^2/m^2, i.e. the understeer gradient that the kinematic bicycle model ignores (the steering angle needed for
  a given yaw rate rises with v^2 on an understeering car, and the kinematic model attributes all of it to curvature). So
  yaw, curvature and lateral acceleration are 1.5-2x too large on highway windows of every steering-derived set (`ccs`,
  `ccs_honda`, `comma2k19`) relative to the lab's IMU / pose yaw. Two qualifications, added on 11 Nov: the logged
  `yawRate` is not a sensor but 2018 openpilot's own single-track vehicle model (`UNDERSTEER.md` section 9: reproduced
  from the same steering angle and speed to a relative error of 3e-8), so the table above compares two models, the
  kinematic one against one with an understeer term, and says that the 2018 model carried a k of 0.0011-0.0014 for these
  two cars; the evidence that the lab's models see the same bias is the lab's own IMU / pose yaw, on which the same
  1 + k v^2 shape appears on 78 platforms (`UNDERSTEER.md` section 7). This bounds the lateral signal's transfer from
  below (3.6 point 6), is a candidate explanation of the yaw-mask sensitivity and of the full model's size loss on `ccs`
  (2.12), and was not visible there because a logged yaw exists on Ford only. The remedy, an understeer-corrected derived
  yaw (divide by 1 + k v^2 with one global k fitted on the lab's data) and a re-decode of the three sets, was run after
  this section was written: 2.15; on these two cars it changes no car-level decision.

Caveats: two cars, one driver each, a commute on one Californian highway in 2018, 2018 openpilot software (its own
longitudinal control in the engaged minutes, old speed-filter gains, a transmit-echo bus layout), so the decisions are probes
and the automation split is an observation, not a fleet statement; `CarParams` constructed from the live fingerprint (no
logged one to compare), and the 2018 safety and vehicle parameters may differ from the constructed ones; the HUD and
`PRE_COLLISION` waiver; the two shared artefacts of 2.13 (sample 0 all-zero in every window; oncoming traffic through the lead
gate on < 0.1 % of the valid lead samples at relative speed below -30 m/s, 0.3 % at lead speed below -2 m/s, minimum
-62 m/s); the understeer bias of the derived yaw (above) applies to this set itself, and its yaw check compares two models
(the logged `yawRate` is 2018 openpilot's vehicle-model output); the 2018 `aEgo` and the decoder's
differ (the models see the decoder's, which is the training convention); the two devices carry `ccs_v_` pseudonyms like
every other device and are outside the phase-3 pooled scope; MIT-licensed data, public platform names only.

### 2.15 Understeer-corrected yaw re-decode (`ccs_us`, `ccs_honda_us`, `comma2k19_us`)

What it is: the three steering-derived sets of 2.12-2.14 decoded a second time with the decoder's bicycle-model yaw divided by
the linear single-track steady-state factor 1 + k v^2 (`ccs_decode.py --yaw derived-understeer --understeer-k 0.00079289`;
curvature and lateral acceleration follow from the corrected yaw; `CCS_PIPELINE.md` "Understeer-corrected yaw"), every other
option exactly as the original decode of each set (the same lists, the Honda gearbox option on `ccs_honda_us`, the three
comma2k19 options on `comma2k19_us`), then scored with the same nine runs. The finding behind it is 2.14's: the kinematic
bicycle model has no understeer, so its yaw grows relative to the true yaw as 1 + k v^2, about 2x at 30-40 m/s. The fit of k
(`understeer_fit.py`, `UNDERSTEER.md`, `understeer_fit.json`) uses the lab's own released drive tables, where the IMU / pose yaw
the models were trained on is the truth: 7,472 drives read, 6,657 usable, 225 platforms with samples, 78 eligible (at least 2
vehicles, 1 h and 1,000 samples), 62 of them train / validation platforms with 423.8 h; per platform a one-parameter least
squares of yaw_b = (1 + k1 v^2) yaw_meas after a per-drive straight-driving angle offset and one robust pass (R^2 0.91-1.00).
The global k is the hours-weighted median of k1 over the 62 train / validation platforms, 0.00079289 s^2/m^2 (characteristic
speed 1 / sqrt(k) = 35.5 m/s; plain median 0.000837, interquartile range 0.000690-0.001223; the two-parameter and speed-bin
estimators agree within 5.2 %, the angle variant and the human-only / controller-on subsets move it by at most 3.7 %). ONE
value for every platform, by design: a per-platform k from the lab's yaw would carry a platform-specific lab calibration into
the external features and would exist for the seen platforms only; the global constant removes the speed-dependent bias every
car shares and leaves the platform residual in the external data. k does not relate to footprint or class (Spearman -0.11 /
-0.26; the class medians sit inside each other's interquartile ranges), so the correction carries no class information. The
two-parameter fits also show the logged `carParams` geometry (steerRatio / wheelbase / angle scale) off by more than 10 % on a
third of the platforms (a > 1.1 on 32.1 %, a < 0.9 on 10.3 %), a speed-independent error the correction cannot fix and the
external decodes inherit. Validation on comma2k19 against the logged 2018 `yawRate` (itself a vehicle-model output, 2.14): the
RAV4 ratios go from 0.98-2.06 to 0.97-1.15 across the speed bins, the Civic's from 0.86-2.01 to 0.85-1.13; a residual of
about +15 % remains at 30-40 m/s and the Civic's 0.86 at 1-5 m/s is geometry, not understeer. The collector (`collect_scores.py`,
section "Understeer-corrected yaw re-decodes" at the end of `RESULTS_TABLES.md`; `results_summary.json["yaw_correction"]`)
first checks the alignment of each pair (same `vehicle_id` / `drive_id` / `w_idx` rows in the same order, same label columns,
every non-yaw channel with max |diff| 0 in float32 and the same NaN pattern, the yaw group changed, the meta columns equal, the
notes equal after the `yaw=` / `k=` fields): OK on all three (45,763 / 2,132 / 381 windows; max |diff| of the yaw group 12.2
rad/s / 0.175 1/m / 853 m/s^2 on `ccs`: a handful of samples with extreme steering angles at speed, which the correction
shrinks most; the largest sits at a logged 69.7 m/s with a 478 deg steering angle, a CAN glitch both decodes carry unchanged in
every other channel, and 15 windows have |yaw| above 2 rad/s before the correction, 23 a lateral acceleration above 30 m/s^2). The `*_us` sets are registered with the kind of the `ccs` case and kept out of every pre-existing table: the per-dataset blocks, the summary table,
the named-vehicle rows, every pooled-decision scope (0, 3.1, 3.2) and the automation split, so nothing above changes;
`ccs_selftest.py` passes with the understeer pair included. Decode times 2.9 h / 10 min / 3 min with 6 workers.

(a) Full model and V1 at device and model level per stratum, both tasks, original -> corrected (delta in points; devices right
per class and models right per class before -> after / support; condensed from the collector section, window level in the
note below; internal references for the full model: size vehicle 63.5 / 59.7 (SMALL 79.2, MID 47.3, LARGE 50.6 recall), drive
70.8 / 67.3; powertrain vehicle 93.9 / 93.4). Answers: how much of the full model's internal size number the corrected yaw
restores in each stratum, and whether the powertrain result moves.

| set | run | task | stratum | device acc % before -> after (delta) | device macro-F1 before -> after | devices right per class before -> after / support | models right before -> after (of N) | models right per class before -> after / support |
|---|---|---|---|---|---|---|---|---|
| `ccs` | full | size | all | 54.1 -> 63.2 (+9.1) | 46.4 -> 60.3 | SMALL 1,606 -> 1,594 / 1,762; MID 351 -> 442 / 1,274; LARGE 241 -> 532 / 1,025 | 63 -> 72 (113) | SMALL 49 -> 47 / 50; MID 8 -> 8 / 33; LARGE 6 -> 17 / 30 |
| `ccs` | full | size | seen | 53.9 -> 64.3 (+10.4) | 47.3 -> 61.8 | SMALL 1,220 -> 1,221 / 1,343; MID 281 -> 343 / 992; LARGE 223 -> 494 / 864 | 40 -> 47 (70) | SMALL 31 -> 30 / 32; MID 4 -> 4 / 20; LARGE 5 -> 13 / 18 |
| `ccs` | full | size | held-out | 53.8 -> 58.0 (+4.2) | 41.6 -> 50.5 | SMALL 368 -> 356 / 404; MID 75 -> 102 / 293; LARGE 22 -> 43 / 167 | 13 -> 14 (29) | SMALL 8 -> 7 / 8; MID 4 -> 4 / 12; LARGE 1 -> 3 / 9 |
| `ccs` | full | size | absent | 88.9 -> 88.9 (+0.0) | 31.4 -> 48.0 | SMALL 32 -> 31 / 32; MID 0 -> 0 / 1; LARGE 0 -> 1 / 3 | 10 -> 11 (14) | SMALL 10 -> 10 / 10; MID 0 -> 0 / 1; LARGE 0 -> 1 / 3 |
| `ccs` | V1 | size | all | 48.2 -> 59.0 (+10.9) | 44.2 -> 57.8 | SMALL 1,069 -> 1,320 / 1,762; MID 695 -> 649 / 1,274; LARGE 193 -> 429 / 1,025 | 61 -> 68 (113) | SMALL 39 -> 44 / 50; MID 18 -> 12 / 33; LARGE 4 -> 12 / 30 |
| `ccs` | V1 | size | seen | 48.1 -> 59.7 (+11.6) | 45.1 -> 59.1 | SMALL 817 -> 1,015 / 1,343; MID 538 -> 490 / 992; LARGE 185 -> 405 / 864 | 38 -> 47 (70) | SMALL 24 -> 29 / 32; MID 10 -> 7 / 20; LARGE 4 -> 11 / 18 |
| `ccs` | V1 | size | held-out | 47.8 -> 55.8 (+8.0) | 38.8 -> 48.7 | SMALL 236 -> 290 / 404; MID 165 -> 163 / 293; LARGE 12 -> 29 / 167 | 14 -> 12 (29) | SMALL 6 -> 6 / 8; MID 8 -> 5 / 12; LARGE 0 -> 1 / 9 |
| `ccs` | V1 | size | absent | 75.0 -> 75.0 (+0.0) | 29.0 -> 29.0 | SMALL 27 -> 27 / 32; MID 0 -> 0 / 1; LARGE 0 -> 0 / 3 | 9 -> 9 (14) | SMALL 9 -> 9 / 10; MID 0 -> 0 / 1; LARGE 0 -> 0 / 3 |
| `ccs` | full | powertrain | all | 87.8 -> 88.9 (+1.1) | 84.7 -> 85.9 | BEV 538 -> 539 / 621; ICE 1,667 -> 1,693 / 1,891 | 86 -> 85 (88) | BEV 14 -> 13 / 15; ICE 72 -> 72 / 73 |
| `ccs` | full | powertrain | seen | 86.1 -> 87.4 (+1.3) | 83.5 -> 84.8 | BEV 471 -> 471 / 545; ICE 1,287 -> 1,313 / 1,496 | 50 -> 50 (51) | BEV 8 -> 8 / 9; ICE 42 -> 42 / 42 |
| `ccs` | full | powertrain | held-out | 94.2 -> 94.4 (+0.2) | 90.0 -> 90.5 | BEV 69 -> 70 / 80; ICE 367 -> 367 / 383 | 27 -> 26 (27) | BEV 6 -> 5 / 6; ICE 21 -> 21 / 21 |
| `ccs` | full | powertrain | absent | 95.5 -> 95.5 (+0.0) | - | ICE 21 -> 21 / 22 | 9 -> 9 (10) | ICE 9 -> 9 / 10 |
| `ccs` | V1 | powertrain | all | 76.4 -> 84.9 (+8.5) | 74.0 -> 82.0 | BEV 577 -> 564 / 621; ICE 1,343 -> 1,569 / 1,891 | 64 -> 71 (88) | BEV 15 -> 13 / 15; ICE 49 -> 58 / 73 |
| `ccs` | V1 | powertrain | seen | 79.0 -> 87.6 (+8.6) | 77.1 -> 85.5 | BEV 514 -> 508 / 545; ICE 1,098 -> 1,280 / 1,496 | 41 -> 47 (51) | BEV 9 -> 9 / 9; ICE 32 -> 38 / 42 |
| `ccs` | V1 | powertrain | held-out | 66.3 -> 73.4 (+7.1) | 60.8 -> 65.7 | BEV 67 -> 60 / 80; ICE 240 -> 280 / 383 | 17 -> 17 (27) | BEV 6 -> 4 / 6; ICE 11 -> 13 / 21 |
| `ccs` | V1 | powertrain | absent | 59.1 -> 77.3 (+18.2) | - | ICE 13 -> 17 / 22 | 6 -> 7 (10) | ICE 6 -> 7 / 10 |
| `ccs_honda` | full | size | all / seen / held-out / absent | 98.3 -> 93.5 (-4.8) / 98.2 -> 93.3 / 100.0 -> 100.0 / 100.0 -> 100.0 | - (one class) | SMALL 226 -> 215 / 230 (seen 220 -> 209 / 224; HR-V 5 -> 5 / 5; Freed 1 -> 1 / 1) | 5 -> 5 (5) | SMALL 5 -> 5 / 5 |
| `ccs_honda` | V1 | size | all / seen / held-out / absent | 47.8 -> 66.1 (+18.3) / 48.2 -> 65.6 / 20.0 -> 80.0 / 100.0 -> 100.0 | - | SMALL 110 -> 152 / 230 (seen 108 -> 147 / 224; HR-V 1 -> 4 / 5; Freed 1 -> 1 / 1) | 3 -> 4 (5) | SMALL 3 -> 4 / 5 (the HR-V recovered) |
| `ccs_honda` | full | powertrain | all / seen / held-out | 99.1 -> 99.6 (+0.4) / 99.1 -> 99.6 / 100.0 -> 100.0 | - | ICE 227 -> 228 / 229 (seen 222 -> 223 / 224; HR-V 5 -> 5 / 5) | 4 -> 4 (4) | ICE 4 -> 4 / 4 |
| `ccs_honda` | V1 | powertrain | all / seen / held-out | 86.0 -> 96.1 (+10.0) / 86.6 -> 96.0 / 60.0 -> 100.0 | - | ICE 197 -> 220 / 229 (seen 194 -> 215 / 224; HR-V 3 -> 5 / 5) | 4 -> 4 (4) | ICE 4 -> 4 / 4 |
| `comma2k19` | full | size | all (= seen) | 50.0 -> 50.0 (+0.0) | - | SMALL 1 -> 1 / 2 (RAV4 SMALL 0.54 -> 0.59; Civic MID 0.491 -> 0.507 against SMALL 0.485 -> 0.434) | 1 -> 1 (2) | SMALL 1 -> 1 / 2 |
| `comma2k19` | V1 | size | all | 0.0 -> 100.0 (+100.0) | - | SMALL 0 -> 2 / 2 | 0 -> 2 (2) | SMALL 0 -> 2 / 2 |
| `comma2k19` | full | powertrain | all | 100.0 -> 100.0 (+0.0) | - | ICE 2 -> 2 / 2 (P(ICE) 0.68 / 0.71 -> 0.73 / 0.74, RAV4 / Civic) | 2 -> 2 (2) | ICE 2 -> 2 / 2 |
| `comma2k19` | V1 | powertrain | all | 100.0 -> 100.0 (+0.0) | - | ICE 2 -> 2 / 2 | 2 -> 2 (2) | ICE 2 -> 2 / 2 |

Window level (all windows): `ccs` full size 49.6 -> 53.8 % / macro-F1 43.2 -> 51.0 (SMALL recall 78.1 -> 74.5, MID 32.6 -> 36.8,
LARGE 20.1 -> 39.0), V1 size 47.5 -> 54.1 / 42.3 -> 51.1, full powertrain 81.8 -> 83.6 / 74.7 -> 76.4, V1 powertrain 63.1 -> 71.1;
`ccs_honda` full size 83.8 -> 81.0, V1 size 49.6 -> 59.3, full powertrain 89.3 -> 90.5, V1 72.7 -> 80.6; `comma2k19` full size
51.7 -> 56.7 (RAV4 55.7 -> 74.6, Civic 48.0 -> 39.8), V1 size 44.4 -> 48.0, full powertrain 70.9 -> 78.0, V1 60.4 -> 89.5. By
mean window speed on `ccs` (full model, window level, sample 0 excluded from the mean): the LARGE recall goes from 34.7 to 38.2 %
at 0-5 m/s, 35.2 -> 45.5 at 5-10, 29.0 -> 45.0 at 10-15, 20.7 -> 42.1 at 15-20, 13.2 -> 40.6 at 20-25, 7.0 -> 31.1 at 25-30 and
4.2 -> 28.0 at 30-45 m/s, i.e. the gain grows with the size of the correction (factor 1 / (1 + k v^2) = 0.96 at 7.5 m/s, 0.71 at
22.5, 0.51 at 35); the SMALL recall gives back 2.0-6.5 points in every bin but 25-30 m/s (+0.6) and MID moves by -2.4 to +9.5.

(b) Platform decision flips of the full model (mean probability over the platform's labelled windows -> arg-max; all 27 size
flips of `ccs` and the one powertrain flip; footprint from the window meta, `*` = borderline; `recovered` = miss -> hit,
`lost` = hit -> miss). Answers: which platforms the corrected yaw brings back and which it loses, and at what confidence.

| platform | model | footprint m^2 | class | stratum | decision before -> after | P(true) before -> after | devices right before -> after / devices | kind |
|---|---|---|---|---|---|---|---|---|
| CHRYSLER_PACIFICA_2018 | Chrysler Pacifica | 5.36 | LARGE | seen | SMALL -> LARGE | 0.22 -> 0.46 | 2 -> 15 / 18 | recovered |
| CHRYSLER_PACIFICA_2018_HYBRID | Chrysler Pacifica | 5.36 | LARGE | seen | SMALL -> LARGE | 0.17 -> 0.39 | 0 -> 16 / 29 | recovered |
| CHRYSLER_PACIFICA_2019_HYBRID | Chrysler Pacifica | 5.36 | LARGE | seen | SMALL -> LARGE | 0.20 -> 0.43 | 18 -> 55 / 96 | recovered |
| CHRYSLER_PACIFICA_2020 | Chrysler Pacifica | 5.36 | LARGE | seen | SMALL -> LARGE | 0.24 -> 0.49 | 7 -> 29 / 41 | recovered |
| HYUNDAI_PALISADE | Hyundai Palisade | 4.96 | LARGE | seen | SMALL -> LARGE | 0.30 -> 0.47 | 24 -> 46 / 63 | recovered |
| MAZDA_CX9_2021 | Mazda CX-9 | 5.15 | LARGE | seen | SMALL -> LARGE | 0.36 -> 0.49 | 4 -> 10 / 13 | recovered |
| TOYOTA_SIENNA_4TH_GEN | Toyota Sienna | 5.30 | LARGE | held-out | SMALL -> LARGE | 0.34 -> 0.49 | 15 -> 27 / 32 | recovered |
| RIVIAN_R1 | Rivian R1T/R1S | 5.58 | LARGE | seen | SMALL -> LARGE | 0.33 -> 0.49 | 39 -> 62 / 96 | recovered |
| FORD_MAVERICK_MK1 | Ford Maverick | 4.93* | LARGE | seen | SMALL -> LARGE | 0.21 -> 0.40 | 2 -> 19 / 32 | recovered |
| FORD_RANGER_MK2 | Ford Ranger | 5.30 | LARGE | held-out | SMALL -> LARGE | 0.38 -> 0.55 | 2 -> 3 / 3 | recovered |
| GENESIS_G80 | Genesis G80 | 4.95 | LARGE | held-out | SMALL -> LARGE | 0.26 -> 0.46 | 1 -> 7 / 7 | recovered |
| HYUNDAI_GENESIS | Hyundai Genesis | 4.90* | LARGE | seen | MID -> LARGE | 0.24 -> 0.47 | 2 -> 9 / 13 | recovered |
| CADILLAC_ESCALADE_ESV_2019 | Cadillac Escalade ESV | 5.76 | LARGE | seen | SMALL -> LARGE | 0.08 -> 0.72 | 0 -> 1 / 1 | recovered |
| CADILLAC_ESCALADE | Cadillac Escalade | 5.14 | LARGE | absent | SMALL -> LARGE | 0.19 -> 0.96 | 0 -> 1 / 1 | recovered |
| VOLKSWAGEN_CRAFTER_MK2 | Volkswagen Crafter | 6.24 | LARGE | seen | SMALL -> LARGE | 0.28 -> 0.56 | 0 -> 3 / 4 | recovered |
| HONDA_ACCORD | Honda Accord | 4.54 | MID | seen | SMALL -> MID | 0.45 -> 0.44 | 55 -> 61 / 120 | recovered |
| HYUNDAI_IONIQ_6 | Hyundai Ioniq 6 | 4.80 | MID | seen | SMALL -> MID | 0.43 -> 0.47 | 7 -> 9 / 24 | recovered |
| HYUNDAI_SANTA_FE_HEV_2022 | Hyundai Santa Fe | 4.57 | MID | held-out | SMALL -> MID | 0.42 -> 0.47 | 4 -> 6 / 8 | recovered |
| KIA_SORENTO | Kia Sorento | 4.54 | MID | held-out | SMALL -> MID | 0.45 -> 0.51 | 0 -> 1 / 1 | recovered |
| FORD_MUSTANG_MACH_E_MK1 | Ford Mustang Mach-E | 4.84* | MID | seen | MID -> SMALL | 0.46 -> 0.35 | 7 -> 5 / 12 | lost |
| TESLA_MODEL_3 | Tesla Model 3 | 4.57 | MID | seen | MID -> SMALL | 0.53 -> 0.46 | 17 -> 11 / 28 | lost |
| JEEP_GRAND_CHEROKEE | Jeep Grand Cherokee | 4.41* | SMALL | held-out | SMALL -> MID | 0.50 -> 0.37 | 16 -> 7 / 21 | lost |
| JEEP_GRAND_CHEROKEE_2019 | Jeep Grand Cherokee | 4.41* | SMALL | held-out | SMALL -> MID | 0.47 -> 0.34 | 10 -> 4 / 15 | lost |
| ACURA_ILX | Acura ILX | 4.06 | SMALL | seen | SMALL -> LARGE | 0.40 -> 0.27 | 1 -> 0 / 1 | lost (4 windows) |
| FORD_ESCAPE_MK4_5 | Ford Escape | 4.39* | SMALL | seen | SMALL -> MID | 0.58 -> 0.38 | 1 -> 0 / 1 | lost (1 window) |
| HONDA_PILOT | Honda Pilot | 4.91* | LARGE | seen | SMALL -> MID | 0.09 -> 0.19 | 2 -> 5 / 77 | wrong -> wrong |
| KIA_CARNIVAL_4TH_GEN | Kia Carnival | 5.30 | LARGE | held-out | SMALL -> MID | 0.07 -> 0.12 | 0 -> 0 / 15 | wrong -> wrong |
| FORD_F_150_LIGHTNING_MK1 (powertrain) | Ford F-150 Lightning | 6.41 | BEV | held-out | BEV -> ICE | 0.52 -> 0.46 | 4 -> 4 / 6 | lost (the only powertrain flip of 108 platforms) |

Platforms right 95 -> 108 of 161 for size (19 recovered, 6 lost, 2 wrong -> wrong), 106 -> 105 of 108 for powertrain. LARGE
platforms still missed after the correction, with devices right after / devices and P(LARGE): Honda Odyssey 11 / 96 (0.18),
Honda Pilot 5 / 77 (0.19), Honda Ridgeline 26 / 56 (0.34), Ford Explorer 23 / 44 (0.38), Toyota Sienna (older platform) 1 / 62
(0.12), Kia Carnival 0 / 15 (0.12), Volkswagen Atlas 1 / 32 (0.19), Genesis G90 0 / 9 (0.11), GV80 0 / 3 (0.12), Santa Cruz 0 / 1,
Transporter 0 / 2, Staria 0 / 1, Durango 0 / 1. MID platforms recovered at device level without a model flip: Lexus ES TSS2
21 -> 33 / 62, Highlander TSS2 37 -> 63 / 164, Sonata 2 -> 9 / 37 and Sonata Hybrid 1 -> 9 / 13, Lexus RX 4 -> 11 / 55 and RX
TSS2 3 -> 12 / 69, Ioniq 5 46 -> 57 / 101 (already MID); the Camry (4 -> 1 / 86, 8 -> 5 / 81), Ascent 0 / 34, RDX 0 / 18,
Tucson 3 / 27, Sorento 4th gen 0 / 16 and the Avalon platforms (1 -> 3 / 37) stay SMALL.

(c) Devices right per class before -> after, `ccs`, full size model (device confusion, rows = true class, columns = decision;
device-level recall in brackets, internal vehicle-level recall of the lab's held-out test for comparison). Answers: where the
nine recovered points come from and what each class still lacks against the internal test.

| true class | devices | called SMALL before -> after | called MID before -> after | called LARGE before -> after | recall % before -> after | internal vehicle recall % |
|---|---|---|---|---|---|---|
| SMALL | 1,762 | 1,606 -> 1,594 | 149 -> 137 | 7 -> 31 | 91.1 -> 90.5 | 79.2 |
| MID | 1,274 | 918 -> 809 | 351 -> 442 | 5 -> 23 | 27.6 -> 34.7 | 47.3 |
| LARGE | 1,025 | 607 -> 335 | 177 -> 158 | 241 -> 532 | 23.5 -> 51.9 | 50.6 |
| all | 4,061 | - | - | - | 54.1 -> 63.2 (macro-F1 46.4 -> 60.3) | 63.5 (59.7) |

By automation state (device decided from its windows of that state): human-driven minutes 55.7 -> 63.6 % of 3,486 devices,
engaged minutes 48.3 -> 56.3 % of 2,700 (window accuracy 51.0 -> 54.6 and 47.6 -> 52.6); the correction lifts both states by the
same eight points, so the engaged-minute deficit of 2.12 (e) is not the understeer term.

(d) The held-out stratum of `ccs` model by model, full size model, original -> corrected (`us_heldout_compare.py`, its output in
`us_heldout_compare.md`; device = mean probability over the device's labelled windows -> arg-max, model = mean probability over
the model's windows -> arg-max, truth = majority label; 871 device x model pairs, 7 devices appearing under two held-out
models, against 864 devices in the stratum rows of (a); `internal` = the lab's own test windows of that model,
`internal_per_model.md`: drive accuracy / decision (P true)). Answers: which held-out models the corrected yaw repairs, which it
breaks, and whether the external replica now agrees with the lab's held-out test model by model.

| held-out model | label | devices | windows | devices right (acc %) before -> after | delta (points) | model decision (P true) before -> after | flip | internal: drive acc / decision (P true) |
|---|---|---|---|---|---|---|---|---|
| Chevrolet Equinox | SMALL | 5 | 243 | 5/5 (100.0) -> 5/5 (100.0) | +0.0 | SMALL (0.64) -> SMALL (0.63) OK | same | 50 / SMALL (0.37) |
| Ford F-150 Lightning | LARGE | 6 | 113 | 5/6 (83.3) -> 6/6 (100.0) | +16.7 | LARGE (0.55) -> LARGE (0.75) OK | same | 95 / LARGE (0.73) |
| Ford Ranger | LARGE | 3 | 82 | 2/3 (66.7) -> 3/3 (100.0) | +33.3 | SMALL (0.38) -> LARGE (0.55) OK | recovered | 90 / LARGE (0.74) |
| Genesis G80 | LARGE | 7 | 304 | 1/7 (14.3) -> 7/7 (100.0) | +85.7 | SMALL (0.26) -> LARGE (0.46) OK | recovered | 0 / MID (0.13) miss |
| Genesis G90 | LARGE | 9 | 249 | 0/9 (0.0) -> 0/9 (0.0) | +0.0 | SMALL (0.06) -> SMALL (0.11) miss | same | 0 / MID (0.13) miss |
| Genesis GV60 | MID | 2 | 97 | 1/2 (50.0) -> 1/2 (50.0) | +0.0 | MID (0.52) -> MID (0.46) OK | same | 97 / MID (0.61) |
| Genesis GV70 | MID | 8 | 200 | 1/8 (12.5) -> 3/8 (37.5) | +25.0 | SMALL (0.36) -> SMALL (0.39) miss | same | 67 / MID (0.45) |
| Genesis GV80 | LARGE | 3 | 146 | 0/3 (0.0) -> 0/3 (0.0) | +0.0 | SMALL (0.06) -> SMALL (0.12) miss | same | 62 / LARGE (0.51) |
| Hyundai Azera | MID | 1 | 5 | 0/1 (0.0) -> 0/1 (0.0) | +0.0 | SMALL (0.32) -> SMALL (0.30) miss | same | 80 / MID (0.50) |
| Hyundai Ioniq | SMALL | 26 | 935 | 26/26 (100.0) -> 26/26 (100.0) | +0.0 | SMALL (0.70) -> SMALL (0.71) OK | same | 69 / SMALL (0.59) |
| Hyundai Kona | SMALL | 20 | 538 | 18/20 (90.0) -> 19/20 (95.0) | +5.0 | SMALL (0.63) -> SMALL (0.64) OK | same | 86 / SMALL (0.60) |
| Hyundai Santa Cruz | LARGE | 1 | 49 | 0/1 (0.0) -> 0/1 (0.0) | +0.0 | SMALL (0.09) -> SMALL (0.17) miss | same | 83 / LARGE (0.45) |
| Hyundai Santa Fe | MID | 62 | 2355 | 39/62 (62.9) -> 48/62 (77.4) | +14.5 | MID (0.46) -> MID (0.48) OK | same | 66 / MID (0.43) |
| Jeep Grand Cherokee | SMALL | 36 | 1438 | 26/36 (72.2) -> 11/36 (30.6) | -41.7 | SMALL (0.49) -> MID (0.35) miss | lost | 22 / LARGE (0.23) miss |
| Kia Carnival | LARGE | 15 | 477 | 0/15 (0.0) -> 0/15 (0.0) | +0.0 | SMALL (0.07) -> MID (0.12) miss | wrong->wrong | 77 / LARGE (0.50) |
| Kia K5 | MID | 3 | 99 | 0/3 (0.0) -> 0/3 (0.0) | +0.0 | SMALL (0.26) -> SMALL (0.30) miss | same | 25 / SMALL (0.40) miss |
| Kia Optima | MID | 5 | 163 | 0/5 (0.0) -> 0/5 (0.0) | +0.0 | SMALL (0.30) -> SMALL (0.25) miss | same | 85 / MID (0.56) |
| Kia Sorento | MID | 17 | 545 | 0/17 (0.0) -> 1/17 (5.9) | +5.9 | SMALL (0.26) -> SMALL (0.25) miss | same | 57 / SMALL (0.40) miss |
| Kia Stinger | MID | 10 | 479 | 10/10 (100.0) -> 10/10 (100.0) | +0.0 | MID (0.56) -> MID (0.56) OK | same | 0 / SMALL (0.33) miss |
| Lexus RX | MID | 124 | 1931 | 7/124 (5.6) -> 23/124 (18.5) | +12.9 | SMALL (0.32) -> SMALL (0.37) miss | same | 17 / SMALL (0.32) miss |
| Nissan Altima | MID | 4 | 193 | 0/4 (0.0) -> 0/4 (0.0) | +0.0 | SMALL (0.32) -> SMALL (0.27) miss | same | 0 / SMALL (0.20) miss |
| Nissan Leaf | SMALL | 28 | 916 | 17/28 (60.7) -> 16/28 (57.1) | -3.6 | SMALL (0.54) -> SMALL (0.50) OK | same | 61 / SMALL (0.56) |
| Tesla Model Y | MID | 21 | 538 | 17/21 (81.0) -> 14/21 (66.7) | -14.3 | MID (0.59) -> MID (0.51) OK | same | 66 / MID (0.45) |
| Toyota Avalon | MID | 37 | 1250 | 1/37 (2.7) -> 3/37 (8.1) | +5.4 | SMALL (0.29) -> SMALL (0.31) miss | same | 50 / SMALL (0.44) miss |
| Toyota Corolla | SMALL | 272 | 1952 | 259/272 (95.2) -> 262/272 (96.3) | +1.1 | SMALL (0.65) -> SMALL (0.64) OK | same | 95 / SMALL (0.72) |
| Toyota Sienna | LARGE | 94 | 1854 | 15/94 (16.0) -> 28/94 (29.8) | +13.8 | SMALL (0.19) -> SMALL (0.30) miss | same | 45 / LARGE (0.42) |
| Volkswagen Atlas | LARGE | 32 | 961 | 0/32 (0.0) -> 1/32 (3.1) | +3.1 | SMALL (0.09) -> SMALL (0.19) miss | same | 0 / SMALL (0.14) miss |
| Volkswagen Jetta | SMALL | 12 | 342 | 12/12 (100.0) -> 12/12 (100.0) | +0.0 | SMALL (0.72) -> SMALL (0.65) OK | same | 82 / SMALL (0.65) |
| Volkswagen Passat | SMALL | 8 | 256 | 8/8 (100.0) -> 8/8 (100.0) | +0.0 | SMALL (0.68) -> SMALL (0.59) OK | same | 81 / SMALL (0.56) |

Held-out stratum: devices right 470 -> 507 of 871 pairs (54.0 -> 58.2 %), models right 13 -> 14 of 29 (recovered the Ranger and
the G80, lost the Jeep Grand Cherokee); the two tests agree on 20 of 29 models (12 hits, 8 misses) against 19 before: the G80
and the Stinger are right here and wrong internally, the Grand Cherokee wrong in both (LARGE internally, MID here, SMALL by its
4.41 m^2 footprint); the Carnival, Sienna, GV80, Santa Cruz, GV70, Optima and Azera remain right internally and wrong here.

(e) The yaw-masked reference of each pair (the full model with the yaw group masked, both decodes; the mask hides the only
channels that changed, so the two runs must agree). Answers: whether anything but the yaw group changed between the decodes.

| set | task | window acc / F1 % original = corrected | device acc / F1 % original = corrected | model acc / F1 % original = corrected | probabilities equal |
|---|---|---|---|---|---|
| `ccs` | size | 35.6 / 34.8 = 35.6 / 34.8 | 39.8 / 37.5 = 39.8 / 37.5 | 38.1 / 33.3 = 38.1 / 33.3 | yes |
| `ccs` | powertrain | 82.4 / 75.2 = 82.4 / 75.2 | 86.2 / 83.1 = 86.2 / 83.1 | 98.9 / 97.9 = 98.9 / 97.9 | yes |
| `ccs_honda` | size | 20.2 = 20.2 | 14.3 = 14.3 | 0.0 = 0.0 | yes |
| `ccs_honda` | powertrain | 87.5 = 87.5 | 98.7 = 98.7 | 100.0 = 100.0 | yes |
| `comma2k19` | size | 3.4 = 3.4 | 0.0 = 0.0 | 0.0 = 0.0 | yes |
| `comma2k19` | powertrain | 78.0 = 78.0 | 100.0 = 100.0 | 100.0 = 100.0 | yes |

Reading:

* **Nine points of size accuracy come back, to the level of the lab's own held-out test.** Full model on the same 4,061 devices:
  54.1 -> 63.2 % (macro-F1 46.4 -> 60.3) against the internal held-out vehicle accuracy of 63.5 / 59.7; 72 of 113 models
  (63.7 %) against 28 of 41 internally (68.3 %); seen 64.3 %, held-out 58.0 %, absent 88.9 %. The corrected full model is 9.5 points
  above the speed-only V3 on the same windows (53.7 / 50.0), which is the margin the lateral channels add internally (63.5
  against 54.4); the as-run decode had turned that margin into a 0.4-point gain. The gain concentrates where the correction is
  largest (LARGE window recall +23.8 points at 30-45 m/s against +3.5 at 0-5 m/s) and is the same in human-driven and engaged minutes.
* **The platforms that come back are the minivans and three-row SUVs that 2.12 had lost, plus the pickups and sedans the as-run
  decode had read as SMALL.** Recovered LARGE: the four Pacifica platforms (27 -> 115 of 184 devices), Palisade (24 -> 46 / 63),
  CX-9 (4 -> 10 / 13), Sienna 4th gen (15 -> 27 / 32), Rivian R1 (39 -> 62 / 96), Maverick (2 -> 19 / 32), Ranger (2 -> 3 / 3), G80
  (1 -> 7 / 7), Hyundai Genesis (2 -> 9 / 13), both Escalades and the Crafter; the pickups that were already right gain devices
  too (Silverado 31 -> 42 / 52, RAM 1500 57 -> 81 / 99, F-150 25 -> 40 / 42, Lightning 5 -> 6 / 6). LARGE at model level 6 -> 17 of
  30 and 51.9 % of the devices, the internal LARGE recall (50.6). Not recovered: the Honda Odyssey (11 / 96), Pilot (5 / 77,
  SMALL -> MID) and Ridgeline (26 / 56), the Carnival (0 / 15, SMALL -> MID), Atlas (1 / 32), the older Sienna platform (1 / 62),
  the Explorer (23 / 44, P(LARGE) 0.38) and the G90 / GV80 (0 / 12): the Honda LARGE platforms and the Carnival move towards MID
  rather than LARGE, which is the direction of the correction but not far enough with one global k (the lab's own k1 for the
  Odyssey platforms is 0.00020 and 0.00133, `UNDERSTEER.md` section 7, i.e. the geometry of those ports is part of it). The two
  MID losses are the Tesla Model 3 (P(MID) 0.53 -> 0.46, 17 -> 11 / 28 devices) and the Mustang Mach-E (0.46 -> 0.35, 7 -> 5 / 12),
  two of the six BEV platforms that carried the as-run MID hits (2.12), and the Model Y also loses devices (17 -> 14 / 21); the MID
  gains are the Accord (55 -> 61 / 120), Ioniq 6, Santa Fe Hybrid and the one Sorento device, with device gains on the Lexus ES
  TSS2, Highlander TSS2, Sonata and Lexus RX that do not reach a model flip. MID stays at 8 of 33 models and 34.7 % of the
  devices, 12.6 points under the internal recall (47.3): the ICE sedans and crossovers of 2.12 are still SMALL to the model, and
  the SMALL-side losses (Jeep Grand Cherokee, a borderline 4.41 m^2 car called MID with P 0.37; ILX and Escape on 4 and 1
  windows) are the price of moving the SMALL / MID boundary.
* **The residual gap to the internal test is MID and the SMALL / MID boundary, not LARGE.** Device-level recall after the
  correction against the internal vehicle recall: SMALL 90.5 against 79.2, MID 34.7 against 47.3, LARGE 51.9 against 50.6. The
  corrected external fleet is more SMALL-leaning than the lab's test (809 of 1,274 MID devices still called SMALL) and matches it
  on LARGE; macro-F1 60.3 against 59.7.
* **Powertrain is essentially unchanged, as expected for a decision the yaw mask leaves in place.** Devices 87.8 -> 88.9 %
  (+1.1; ICE recall 88.2 -> 89.5, BEV 86.6 -> 86.8), windows 81.8 -> 83.6, models 86 -> 85 of 88: the one flip is the F-150
  Lightning on 6 devices (P(BEV) 0.52 -> 0.46, 4 of 6 devices right both times), a borderline call before and after; the held-out
  stratum 94.2 -> 94.4 % of devices. The speed-only V3 reference (92.6 %) does not use the yaw group and is unaffected; the full
  model stays below it, so the powertrain conclusions of 2.12 and 3.1 stand.
* **The two SMALL-only sets move inside their class, in opposite directions.** `ccs_honda_us`: 226 -> 215 of 230 devices SMALL
  (98.3 -> 93.5 %); the corrected yaw moves 10 devices from SMALL to MID and 2 from SMALL to LARGE, one MID device returning to
  SMALL (device decisions SMALL / MID / LARGE 226 / 3 / 1
  -> 215 / 12 / 3; per platform Civic 32 -> 29 of 33, Civic 2022 35 -> 33 of 36, Civic Bosch 55 -> 53 of 55, CR-V 5G 91 -> 87 of 92,
  the HR-V, CR-V 2016, Fit and Freed unchanged), every platform stays SMALL (P(SMALL) 0.55-0.77) and the loss is on the
  human-driven minutes (window accuracy 91.1 -> 86.0 human, 73.7 -> 74.2 engaged). A smaller yaw at speed reads as a slightly
  larger car to the model, which costs SMALL devices on ports whose lab k1 scatters around the global value (Civic Bosch
  0.00055, Civic 2022 0.00050, CR-V 5G 0.00111), so one k under- or over-corrects each of them. Powertrain 99.1 -> 99.6 %.
  `comma2k19_us`: no car-level decision changes (RAV4 SMALL 0.54 -> 0.59, Civic MID 0.491 -> 0.507 against SMALL 0.485 -> 0.434;
  both ICE, P 0.68 / 0.71 -> 0.73 / 0.74); the RAV4's engaged minutes turn SMALL (P(SMALL) 0.40 -> 0.56; its engaged windows
  28.6 -> 73.3 % SMALL, all its windows 55.7 -> 74.6), the Civic's stay MID (0.60; engaged windows 24.0 -> 17.4 % SMALL, all its
  windows 48.0 -> 39.8); the powertrain window accuracy rises 70.9 -> 78.0,
  the engaged minutes from P(ICE) 0.49 / 0.56 to 0.57 / 0.61.
* **V1 (lateral + speed) moves more than the full model and in the same direction, on every set.** `ccs` size 48.2 -> 59.0 % of
  devices (+10.9; internal V1 62.6), SMALL recall 60.7 -> 74.9, LARGE 18.8 -> 41.9, MID 54.6 -> 50.9 (its MID over-call of 2.12
  recedes: 18 -> 12 MID models, held-out 8 -> 5), models 61 -> 68; V1 powertrain 76.4 -> 84.9 %
  (ICE recall 71.0 -> 83.0, 49 -> 58 ICE models; internal V1 75.8), so part of V1's BEV over-call on this fleet (2.12) was the
  inflated yaw; `ccs_honda_us` V1 size 47.8 -> 66.1 % and powertrain 86.0 -> 96.1 %, the held-out HR-V 1 -> 4 of 5 and 3 -> 5 of 5;
  `comma2k19_us` V1 both cars SMALL instead of MID. A variant that depends on the lateral channels alone is the most sensitive
  to their construction, which is what 3.5 point 3 said in the other direction.
* **What this changes in 2.12 and 3.6.** 2.12's reading that "the lateral size signal that lifts the full model nine vehicle
  points above V3 internally does not reach new devices through a steering-derived yaw" was a statement about the as-run decode:
  with the yaw derived consistently with the training signal (the understeer term the IMU / pose yaw contains and the kinematic
  model lacks) the same signal reaches the same devices, and the full model's size transfer on a fleet of unknown devices is at
  its internal level in aggregate (device accuracy and macro-F1; LARGE at, MID below its internal recall). 3.6 point 6 ("the size result bounds the transfer of the lateral signal from below and does not say whether
  a measured yaw would restore it") is answered for the understeer part: a corrected derived yaw restores it; the "nine points
  lost" of 3.6 point 4 were, on this evidence, a decode bias rather than a model limit. The parts of 2.12 that stand: SMALL transfers, MID is the weak
  class in every stratum, "seen platform, new device" is not easier than a new platform (seen 64.3 against held-out 58.0 after the
  correction; the held-out stratum stays 5.5 points under the lab's held-out test), the engaged-minute and lead-group findings for powertrain,
  and the per-platform systematics of the misses.
* **What remains.** A residual of about +15 % at 30-40 m/s after the global k (comma2k19 validation; the lab's per-platform k1
  spans 0.0002-0.0024 over the 62 train / validation platforms, -0.0002 to 0.0025 over all 78 eligible), the geometry error a of a third of the platforms (steerRatio / wheelbase off by more than 10 % at every
  speed, which the lab's k cannot fix and which a per-platform check of the logged `carParams` against the measured yaw could),
  the Honda LARGE ports and the Carnival that move only to MID, and MID itself, which no decode and no stratum lifts above
  35 % of the devices; the sample-0 and oncoming-lead artefacts of 2.13 are unchanged by the re-decode.

Caveats: the correction is post hoc, decided after the as-run results of 2.12-2.14 were known and motivated by a yaw ratio
measured on an external set (comma2k19), then fitted on the lab's own IMU / pose yaw and never tuned on the external labels (one
global k, no per-platform value; the fit used no external label or score); the as-run numbers of 2.12 / 2.13 / 2.14
remain the primary results and the corrected decode is the recommended pipeline going forward, not a replacement. The k is one
constant for all platforms (the lab's k1 spans 0.0002-0.0024 over the train / validation platforms, -0.0002 to 0.0025 over all
eligible, and its a 0.79-1.24), fitted on 62 train / validation platforms
of the lab (423.8 h; heavy platforms weigh most: Rivian R1, Civic Bosch, F-150, Model 3, Tucson, Ioniq 6), on pose-yaw rows
that pass the drive filters, with a per-drive angle offset the external decoder does not remove (the raw-angle variant gives a
k 3.7 % higher). The comma2k19 validation compares two models (the kinematic one and 2018 openpilot's vehicle model that wrote
`carState.yawRate`), so it shows agreement with that model's understeer rather than with a sensor; the lab fit rests on the IMU /
pose yaw. The platform-level flips at P(true) 0.39-0.49 (Pacifica, Palisade, CX-9, Sienna, Rivian, Maverick, G80, Genesis,
Accord, Ioniq 6, Santa Fe HEV) are decisions near the boundary, as the as-run misses were; the single-window flips (ILX, Escape,
Escalade) carry no weight. The sample-0 all-zero `CarState` and the oncoming-lead artefact persist in the re-decodes (only the
yaw group changed). Decisions, strata, labels and aggregation rules are those of 2.12-2.14; intervals are not recomputed here
(the pooled score files of each `*_us` set carry their own 1,000-resample bootstraps). MIT-licensed data, pseudonymous device
ids and public platform names only.

## 3. Cross-dataset summary

### 3.1 Does the powertrain signature transfer?

Pooled vehicle-level decisions of the primary runs (one physical car = one decision; `RESULTS_TABLES.md`):

| scope | BEV cars right | ICE cars right | all |
|---|---|---|---|
| phase 1: all eight labelled sets (2.1-2.9) | 23 / 27 (85.2 %) | 268 / 284 (94.4 %) | 291 / 311 (93.6 %) |
| phase 1 without VED (264 ICE cars of one source) | 23 / 24 (95.8 %) | 10 / 20 (50.0 %) | 33 / 44 (75.0 %) |
| phase 1, longitudinal / speed-only sets only (V2 / V3 variants) | 20 / 24 (83.3 %) | 268 / 283 (94.7 %) | 288 / 307 (93.8 %) |
| phase 2: commaSteeringControl archives (V3 variant; one archive = one pseudo-vehicle) | 8 / 9 (88.9 %) | 27 / 27 (100.0 %) | 35 / 36 (97.2 %) |
| phase 2: MoreTorque cars (full model) | 0 / 1 (0.0 %) | 11 / 11 (100.0 %) | 11 / 12 (91.7 %) |
| phases 1 and 2 | 31 / 37 (83.8 %) | 306 / 322 (95.0 %) | 337 / 359 (93.9 %) |
| phase 3: commaCarSegments devices, `ccs` (full model; one device x car model = one decision) | 544 / 630 (86.3 %) | 1,688 / 1,917 (88.1 %) | 2,232 / 2,547 (87.6 %) |
| phase 3: the recovered Honda platforms, `ccs_honda` (full model; same units) | - | 228 / 231 (98.7 %) | 228 / 231 (98.7 %) |
| phase 3 pooled (`ccs` + `ccs_honda`) | 544 / 630 (86.3 %) | 1,916 / 2,148 (89.2 %) | 2,460 / 2,778 (88.6 %) |
| comma2k19, two cars on 2018 hardware (full model; one device = one decision; a scope of its own) | - | 2 / 2 (100.0 %) | 2 / 2 (100.0 %) |
| all sets | 575 / 667 (86.2 %) | 2,224 / 2,472 (90.0 %) | 2,799 / 3,139 (89.2 %) |

(Phase-3 units are device x car-model pairs judged against the device's own platform label, 2,778 for powertrain and 4,384
for size over `ccs` + `ccs_honda` (2,547 / 4,152 and 231 / 232), a device under two car models counting once per model and per set; the pooled-decision table of `RESULTS_TABLES.md`
(`collect_scores.py`, `pooled_decisions`) counts the same pairs with the pair's own label and agrees with this table.)

* **BEV side: yes, for every BEV with a car-level sample and a 10 Hz speed.** Tesla Model 3 (seven cars over five sets,
  some possibly the same physical car, P(BEV) 54-99 %), Model S and Model X (two sets each), Audi e-tron, Jaguar I-Pace, VW e-Golf (two sets each), Ioniq 5,
  Polestar 2, both Niro EVs, Bolt EUV and Rivian R1 are called BEV; the only named BEV miss is the Ford Mustang Mach-E
  (UGA, P(BEV) 35.8 %, 83 windows of scripted platoon driving), plus the three VED Leafs, which fail the way the held-out
  Leaf fails internally for the speed-only variant. Drive-level BEV recall of the V2 variant is 93.9 % on OpenACC and
  96.0 % on ULTra-AV, above its internal 78.3 %: ACC-driven BEVs look more BEV-like than our human-driven ones.
* **ICE side: mostly only when a human drives.** The human-driven cars (Fiat 500X: 99.1 % of 429 trips, Polo, Space
  Star, Rexton, MINI, the VED fleet 97.7 % of 264 cars) are ICE, with one exception (the UGA Lexus UX 200 lead car on
  its scripted oscillation profile, P(BEV) 57.8 %); the ICE cars following a lead under adaptive cruise in the platoons
  (A6, X5, Mazda 3, XC40 in both sets; the A-Class in OpenACC, 52-53 % either way in both sets) and the ACC-engaged RAV4
  segment are BEV, and the same cars stay BEV-looking on their few manual laps inside the platoon (A6 76 %, X5 73 %);
  two ACC-driven ICE cars resist, the Audi A8 (leader on cruise control, P(BEV) 21.8 %) and the Ford S-Max (ACC
  follower, P(BEV) 43.0 % in OpenACC, 47.8 % in ULTra-AV). So it is not the `das_enabled` flag that misleads
  the model but the longitudinal profile of car following on a test track as the GNSS chain renders it: smooth,
  persistent decelerations without a pedal signal are what the BEV class looks like to a model trained on regenerative
  braking. Internally the V2 variant keeps 93.0 % ICE drive recall on automation-engaged windows, so this is an external
  distribution shift (controller, sensing), not an internal weakness.
* Levels: drive and vehicle aggregation help where the car is right and cannot repair a systematic miss; window-level
  accuracy on OpenACC (68.2 %) is 8.5 points below the internal V2 window number (76.7), vehicle accuracy (73.7 %) is
  17.7 points below (91.4). Macro-F1 on the only sets with both classes and several cars: OpenACC 73.7 (vehicle) / 76.3
  (drive), ULTra-AV 69.7 / 46.2, UGA 33.3; chance 50.
* Mild hybrids and other out-of-task cars: the 48-V Audi A4 Avant is called BEV on 82 % of its windows in both sets, the
  GLE 450 is split (42-50 %), the RAV4 Hybrid 83 % BEV, the i3s range extender 79 %, the Nexo fuel-cell car 86 %, the
  Outlander PHEV 38 %, the VED hybrids 24 % (HEV) and 23 % (PHEV) of windows: the model places electrified drivetrains
  with regenerative braking on the BEV side, which is consistent with what it was trained to read, and shows that the
  A4 Avant label convention (HEV by the JRC file, ICE by the lab's mild-hybrid rule) matters for an ICE count.
* **Phase 2 settles the ICE question: with the car's own CAN acceleration, ICE cars under automation are ICE.** On the
  36 commaSteeringControl archives (every window lateral-controller-engaged on highways) the speed-only variant calls
  all 27 ICE archives and 8 of 9 BEV archives right (2.10), and on MoreTorque the 11 ICE cars are ICE in every
  automation state (2.11). The phase-1 ICE misses were cars whose acceleration had to be derived from GNSS or OBD
  speeds (3.3), not an effect of the automation itself. The two BEV misses of phase 2 are the Ioniq Electric archive
  (P(BEV) 0.32) and the single Omoda E5 (0.20-0.41 depending on the model); the lateral + speed variant V1, however,
  calls every commaSteeringControl archive ICE, so the lateral channels must stay out of a powertrain model that is
  meant to transfer.

* **Phase 3 puts the number on a fleet: 86 of 88 car models and 87.8 % of 2,512 unknown devices with the full model,
  92.6 % with the speed-only variant (2.12).** The held-out stratum reproduces the lab's held-out vehicle accuracy (94.2
  against 93.9 %); the loss elsewhere is on automation-engaged minutes (window accuracy 64.8 against 92.2 % human-driven,
  the same devices 70.9 against 93.8 %), where the full model's lead group, built from a radar gate or the stock-ACC
  object instead of radard's fused lead, pushes ICE minutes towards BEV (masking it: 81.8 % on engaged minutes, 91.3 %
  overall). So the automation effect of phase 1 has two parts: the derived-acceleration artefact (3.3), which phase 2
  removed, and a genuine engaged-minute drift of the full model on a new fleet, which phase 3 measures and which the
  speed-only inputs largely avoid. Hybrids are read in between (HEV 32.0 %, PHEV 44.6 % of windows BEV, against ICE 17.5 %
  and BEV 78.7 %).
* **The two sets decoded afterwards add ICE evidence only, and one of them shows the engaged-minute drift at its
  strongest.** The nine recovered Honda platforms (2.13) are ICE for 227 of 229 devices (99.1 %) and 4 of 4 models, the
  held-out HR-V 5 of 5, with the two HONDA_CIVIC_2022 misses among the 9 possible-hybrid devices (219 of 220 without them);
  the speed-only V3 is at 98.3 % of the devices, V2 97.8 %, V1 86.0 %. The two comma2k19 cars (2.14) are ICE at car level
  (P(ICE) 0.68 / 0.71) but their engaged minutes, driven by 2018 openpilot's own longitudinal controller, sit at P(BEV)
  0.47 (50.9 % window accuracy against 100.0 % on the human-driven minutes), and the speed-only variant does not avoid it
  there (64.3 % of the windows): on an unseen longitudinal controller the drift is not only the lead group.

### 3.2 Where does size fail, and why?

Pooled vehicle-level size decisions of the phase-1 sets (primary runs): SMALL 25 / 27 cars (92.6 %), MID 2 / 20
(10.0 %), LARGE 0 / 13 (0.0 %); on the longitudinal / speed-only sets alone SMALL 24 / 25, MID 1 / 19, LARGE 0 / 11. The
two MID hits are Tesla Model 3s (training platform: Yuan et al. with V3, the commaCarSegments window with the full
model). Without the seven borderline cars the picture is unchanged (OpenACC vehicle macro-F1 19.4 instead of 18.4;
ULTra-AV 17.9 instead of 17.8). Phase 2 adds, with steering channels present, commaSteeringControl SMALL 15 / 17
archives, MID 4 / 13, LARGE 4 / 6 (V1 variant) and MoreTorque SMALL 8 / 8 cars, MID 0 / 4 (full model); over phases 1 and 2
SMALL 48 / 52, MID 6 / 37, LARGE 4 / 19. Phase 3 (device x car-model pairs, 3.1): `ccs` + `ccs_honda` SMALL 1,863 / 2,030
(91.8 %), MID 360 / 1,301 (27.7 %), LARGE 250 / 1,053 (23.7 %), all 2,473 / 4,384 (56.4 %), of which `ccs_honda` is 228 / 232
SMALL; comma2k19 1 / 2 SMALL; over every set SMALL 1,912 / 2,084 (91.7 %), MID 366 / 1,338 (27.4 %), LARGE 254 / 1,072
(23.7 %), all 2,532 / 4,494 (56.3 %). Points 1-5 below are the phase-1 explanation, point 6 what phase 2 changes, point 7
what phase 3 adds and point 8 what its two addenda add.

1. **The inputs that carry size are missing.** On our own data the size signal lives in the lateral channels (V1 = full
   model: 66.4 vs 67.3 drive macro-F1), and the longitudinal inputs keep little of it (V3 45.1, V2 33.5 = chance). Every
   external set but A2D2 and the commaCarSegments sample lacks steering, yaw and curvature, so at best the weak V3 signal
   was available, and that signal gets MID / LARGE right for 3 / 14 and 3 / 12 of our own held-out models.
2. **What is left is tied to the CAN acceleration estimate and disappears with a derived acceleration** (section 3.3):
   with `a := smooth_derivative(v)` on our own held-out windows the V3 LARGE drive recall falls from 28.7 to 3.4 % and the
   predicted LARGE share from 18.8 to 4.8 %; with GNSS-like smoothing LARGE recall is 0.8 %. The external pattern (P(LARGE)
   below 25 % for every car, P(SMALL) 50-70 % regardless of the car and of the speed regime) is exactly this collapse,
   not a property of the European cars.
3. **LARGE is learned as truck-like.** Where lateral channels exist (A2D2 Q7 e-tron with the full model; Rivian R1 sample
   window) the LARGE cars still fail, and internally the full model misses LARGE sedans and crossovers of unseen makes
   (Model S 6 %, G80 / G90 0 %, Atlas 0 %) while recognising pickups, minivans and three-row SUVs (Lightning 95 %, Yukon
   100 %, Ranger 90 %, Carnival 77 %). The external LARGE cars (Model S / X, I-Pace, X5, A8, GLE 450, Q7) are all of the
   first kind; the one external truck (Rivian) got one window.
4. **Test-track and ACC driving remove the manoeuvres that carry size** (launches, braking to a stop, cornering):
   ZalaZone runs at 30-60 km/h with an ACC gap, AstaZero laps are identical for all cars, the VED and 500X trips are
   urban traffic where the SMALL call is the default (500X 46.2 % of trips SMALL, Leaf 62.5 %).
5. **Borderline labels and one car per model** cannot explain the result (the misses are 0.1-0.6 m^2 away from the cuts
   for the Model S / X, X5, GLE 450, Q7, A6, S-Max, Rexton), but they do mean that no interval can be given per model.
6. **With steering present (phase 2) the size result equals the internal one, and its failures are the internal
   ones.** commaSteeringControl with the V1 variant: 63.9 % archive accuracy / 62.3 macro-F1 against 62.6 / 58.0
   internally, LARGE 4 of 6 (pickups, Pilot, Sienna; Odyssey and Palisade missed, as crossover SUVs are missed
   internally), MID 4 of 13 (internally 43 % vehicle-level recall); the speed-only variant on the same windows gets no
   MID and 3 of 6 LARGE (the pickups and the Pilot from the `aEgo` texture, point 2). MoreTorque with all 20 channels:
   both MID platforms SMALL (Sonata, a training platform; CX-50), both SMALL platforms right; masking the yaw group
   flips the MID cars to LARGE, so the decision rests on the lateral channels, which on engaged highway or fork-steered
   driving do not separate a 4.6-4.7 m^2 sedan or crossover from a SMALL car. The remaining size problem is therefore
   MID (and crossover LARGE) under automation, not the transfer of the signal as such.
7. **On a fleet of unknown devices (phase 3) the lateral advantage of the full model disappears and only the
   `aEgo`-texture part of the size signal is left.** commaCarSegments, full model: SMALL 49 / 50 models, MID 8 / 33, LARGE
   6 / 30 (27.6 and 23.5 % of their devices), 54.1 % of 4,061 devices against 63.5 % internally, the same on the seen (53.9)
   and the held-out (53.8) stratum; the speed-only V3 is at its internal level (53.7 %, macro-F1 50.0 above the full
   model's 46.4) and gets 14 of 30 LARGE models, the full model 6, all pickups. The LARGE minivans and SUVs that the
   internal test recognised (Carnival 77 %, Sienna 45 % of drives) are SMALL on new devices (0 of 15 and 15 of 94
   devices), so point 3 (LARGE is learned as truck-like) is the robust part of the internal result and the minivan hits
   were not. MID is right mostly for BEV platforms (6 of 8 hits: Tesla, E-GMP, Mach-E), which may be the platform signature
   rather than size. The yaw-group mask shows that the full model decides through the lateral channels (69.6 % of the
   devices, 88.3 % of the LARGE ones, turn LARGE without them); since the yaw here is steering-derived and the lateral signature of a car depends
   on device and driver, the set cannot say which of the two breaks the transfer, only that the internal nine-point gain
   of the lateral channels over speed-only inputs does not reach new devices, in no stratum.
8. **The two addenda (2.13, 2.14) confirm the SMALL side and name a candidate cause for the lateral loss.** The recovered
   Honda fleet is SMALL for 226 of 230 devices with the full model and on every platform, while the lateral + speed V1
   calls five of its nine platforms MID and the yaw mask turns 70.7 % of its windows LARGE: the full model's SMALL call
   on small cars runs through the lateral group and holds. On the two 2018 cars the full model's size call splits by
   automation state (human-driven windows SMALL at P 0.68-0.72, engaged ones MID at 0.58-0.61; the Civic ends a coin
   flip), the speed-only variant has both SMALL. And the 2018 logs, which carry a logged yaw rate (2018 openpilot's vehicle-model output, 2.14), show the
   steering-derived yaw of all three CAN sets overestimating it as 1 + k v^2 (1.5x at 20-25 m/s, 2x at 30-40 m/s): the
   lateral channels the full model relies on are inflated on exactly the highway minutes where MID and LARGE are missed,
   a candidate (untested) cause of point 7 and the reason the suggested next step of section 4 is a yaw-corrected
   re-decode rather than a new model.


### 3.3 Diagnostics behind the explanations

**Derived acceleration (`diag_derived_accel.py`, 8,000 random held-out test windows per task, seed ensembles).** The
external adapters have to build `a_mps2` as a smoothed derivative of a GNSS / OBD / held CAN speed; doing the same to our
own windows reproduces the external failure modes:

| variant | input treatment | size: drive acc / macro-F1 (recall S / M / L) | BEV vs ICE: drive acc / macro-F1 (recall BEV / ICE) |
|---|---|---|---|
| V3 speed-only | CAN `aEgo` (as trained) | 52.2 / 45.0 (70.3 / 35.1 / 28.7) | 86.8 / 85.8 (76.5 / 93.4) |
| V3 | `a := smooth_derivative(v)`, 0.5 s (ULTra-AV, OpenACC, microSIM, UGA, A2D2, Tesla) | 43.5 / 32.9 (58.6 / 47.5 / 3.4) | 74.7 / 70.1 (45.1 / 93.8) |
| V3 | derivative, 1.0 s | 46.9 / 32.8 (72.8 / 31.7 / 3.0) | 72.6 / 70.0 (54.9 / 84.0) |
| V3 | speed smoothed 0.5 s + derivative (GNSS-spline-like) | 45.5 / 30.1 (72.8 / 27.9 / 0.8) | 72.9 / 69.4 (49.5 / 88.1) |
| V3 | speed held at 1 Hz + derivative 1.0 s (VED-like) | 48.0 / 29.7 (82.0 / 18.1 / 1.7) | 65.0 / 63.4 (56.5 / 70.4) |
| V2 longitudinal-only | CAN `aEgo` | 44.1 / 34.9 (66.2 / 13.6 / 26.6) | 87.4 / 86.4 (75.9 / 94.9) |
| V2 | derivative, 0.5 s | 43.6 / 31.2 (70.9 / 11.7 / 15.2) | 86.3 / 85.7 (83.8 / 87.9) |
| V2 | derivative, 1.0 s | 43.4 / 32.1 (68.9 / 10.9 / 19.8) | 63.1 / 62.9 (89.5 / 46.0) |
| V2 | GNSS-spline-like | 44.5 / 32.1 (72.1 / 12.1 / 16.0) | 80.5 / 80.2 (86.7 / 76.6) |
| V2 | VED-like | 43.6 / 30.9 (70.7 / 7.2 / 20.7) | 52.7 / 50.9 (91.7 / 27.5) |

Reading: (i) the speed-only size signal is a texture of the 10 Hz `aEgo` (openpilot's speed Kalman filter on each car's
CAN speed), not vehicle dynamics: smoothing it away leaves SMALL. (ii) The longitudinal variant's powertrain decision is
robust to the 0.5-s derivative (its BEV recall even rises) but tips towards BEV as the speed gets smoother (ICE recall
76.6 % GNSS-like, 46.0 % at 1 s, 27.5 % VED-like), which is the direction of the OpenACC / ULTra-AV ICE misses (splined
GNSS speeds, ACC following); the speed-only variant tips the other way (BEV recall 45 %), which is the direction of the
Leaf and Mach-E misses and of the low window-level BEV recall on the Tesla trips (62 %). (iii) The 1 Hz upsampling of VED
costs the most: the speed-only separation falls to 56.5 / 70.4.

**External confirmation (phase 2).** commaSteeringControl carries openpilot's own `aEgo` for 25 models, and the
speed-only variant that collapses under a derived acceleration (BEV drive recall 45 %, LARGE recall 3 % in the table
above) reaches 35 of 36 archives on the powertrain task and 3 of 6 LARGE archives on size there (2.10): the texture of
the car's own acceleration estimate is the signal, as the table predicts. MoreTorque, also with a CAN `aEgo`, calls its
11 ICE cars ICE at 90.6-95.5 % drive accuracy (2.11).

**External confirmation (phase 3).** On commaCarSegments (CAN `aEgo`, 4,061 devices) the speed-only variant reaches its
internal numbers on both tasks (powertrain 92.6 % of devices against 91.0 internal; size 53.7 against 54.4) and gets the
LARGE pickups, the Pacifica, Rivian, Ranger, Durango and Staria from the acceleration texture alone (14 of 30 LARGE
models), while the derived-acceleration sets of phase 1 got none: the texture of the car's own acceleration estimate is
the transferable part of the size signal, and it survives a fleet of unknown devices (2.12).

**Automation state.** Inside the external sets the automation flag itself does not drive the powertrain call (same-car
comparisons in 2.2); pooled ICE recall is 46.4 % on ACC windows and 62.0 % on human windows of OpenACC, BEV recall the
same in both (86 %). Internally the matched variants lose 3-5 points of drive accuracy on automation-engaged windows
(V2 90.0 -> 86.9, V3 89.5 -> 88.3), nothing like the external gap.

Phase 3 separates the two automation effects: on a CAN fleet the full powertrain model still loses 27 points of window
accuracy on engaged minutes (92.2 -> 64.8 %) and 23 points at device level on the same devices (93.8 -> 70.9 %), the
speed-only V3 loses 12 and 10 (86.0 -> 73.5, 94.1 -> 84.3), the lead-masked full model 20 and 13 (93.2 -> 73.3,
94.8 -> 81.8); so beyond the derived-acceleration artefact there is a genuine engaged-minute drift towards BEV that is
strongest in the full model and partly attributable to its lead group as built outside the lab (2.12).

**Consistency with our own held-out models (`internal_per_model.md`).** The external misses that have an internal
counterpart behave the same way: Nissan Leaf (VED 0 / 3 cars BEV with V3; internal V3 18 % of drives, model-level ICE),
Tesla Model S size (external SMALL with V3; internal V3 SMALL 17 % of drives LARGE, full model MID). The external hits
have one too: Tesla Model S powertrain (external BEV 79 %; internal V2 / V3 BEV 89-94 % of drives). The size model that
has the lateral channels gets 28 of 41 held-out models right internally; no external set offered those channels for more
than one car.

### 3.4 The secondary numbers (full model, missing channels at the missing value)

The full 20-channel models were run on every set with the absent channels set to the training "missing" value (NaN ->
standardised 0, `lead_valid` -> 0). On our own test windows this treatment makes the size model call almost every drive
SMALL (SMALL recall 98-99 %, MID 1.4-4.8 %, LARGE 0.4-1.6 %) and the powertrain model call 82-83 % of the BEV drives ICE
(BEV drive recall 17.7 % under the V2 mask, 16.8 % under V3), because the training augmentation only ever dropped the lead, torque or pedal group, never the lateral
channels. The external secondary numbers follow suit (ULTra-AV / OpenACC vehicle BEV recall 0-14 %, Tesla Yuan 0 of 106
trips BEV, VED 0 of 3 Leafs, every LARGE / MID car SMALL, A2D2 SMALL under every mask) and must not be read as transfer
results; the variants trained on the reduced channel lists are the comparison, and even they were trained with the
20-channel recipes unchanged (VARIANTS.md). Phase 2 behaves the same way: the full model under the V1 mask calls every
commaSteeringControl archive SMALL (17 / 17 SMALL, 0 / 13 MID, 0 / 6 LARGE) and every BEV archive ICE (0 / 9), and
under the V1 / V2 / V3 masks on MoreTorque the BEV drive recall is 0.0-2.2 %. Phase 3 (all channels present, so the masks
are pure sensitivity runs): on commaCarSegments the full model under the V2 / V3 masks calls 0 of 15 BEV models BEV and
0 (V2 mask) and 1 (V3 mask: the Genesis G70) of 63 MID / LARGE models right, under the V1 mask 3 of 15 and 1 of 63; the yaw
mask, a state the training never produced, calls 69.6 % of the devices LARGE (88.3 % of the LARGE ones) while leaving the powertrain decision in place (86.2 % of devices, 87 of 88 models); the
lead mask, a group the training did drop, is the one mask that reads as a result (powertrain 91.3 % of devices, size
unchanged).

### 3.5 What the full-channel and lateral sets add (phase 2)

1. **Size transfers when steering is present**, to the level of the internal held-out test: commaSteeringControl V1
   archive accuracy 63.9 % [50.0, 80.6] / macro-F1 62.3 (internal 62.6 / 58.0), model level 16 / 25, held-out stratum
   7 / 10 archives and 5 / 7 models. Phase 1 measured the absence of the inputs, not the absence of the signal.
2. **LARGE and MID on unseen platforms**: LARGE is recognised for the held-out Toyota Sienna and the seen Silverado,
   RAM 1500 and Pilot (4 / 6 archives), not for the Odyssey and the Palisade; MID for the held-out Genesis GV60 and the
   seen Ioniq 5, Tucson and Highlander 2017 (4 / 13), not for the Santa Fe, Accord, Camry, Sonata, Highlander 2020 and
   EV6 archives nor for the MoreTorque Sonata and CX-50 (0 / 4 cars with all 20 channels). MID under highway automation
   is the open problem; the yaw-group mask shows that the models do use the lateral channels for it.
3. **Powertrain with a CAN acceleration**: the speed-only variant transfers across 25 models and two fleets (35 / 36
   archives, 11 / 11 ICE cars), the matched lateral + speed variant does not (0 / 9 BEV archives). For a deployable
   powertrain model the lateral channels should be excluded, or the lead / pedal / torque groups must be present, as in
   the full model (whose internal 93.9 % could not be tested externally without pedals: commaSteeringControl has none).
4. **A seen platform on a new device and fork is not in distribution** (MoreTorque Sonata: MID 0.23 with every
   channel present), which bounds what the `split_platform` numbers promise about a new openpilot fleet and is the
   question the device-disjoint commaCarSegments selection answers at scale in 2.12 / 3.6: 53.9 % size device accuracy on
   70 training models and 3,199 new devices, the same as on the held-out models.
5. **Single-car probes stay probes**: the Omoda E5 (BEV called ICE by every model) and the A2D2 Q7 e-tron (LARGE called
   SMALL) are one car each with fork or format quirks; they are recorded, not generalised.

### 3.6 What the device-disjoint set adds (phase 3)

1. **The powertrain claim holds on a fleet.** 86 of 88 car models and 87.8 % [86.6, 89.1] of 2,512 devices the lab never
   saw with the full 20-channel model, 87 of 88 and 92.6 % [91.5, 93.6] with the speed-only variant; the held-out stratum
   (27 of 27 models, 94.2 % of 463 devices) reproduces the `split_platform` vehicle accuracy (93.9) externally. The
   Nissan Leaf, missed by every speed-only model internally and externally, is BEV for 22 of 28 devices with the full
   model; the F-150 Lightning 4 of 6.
2. **"Seen platform, new device" is not in distribution, for either task.** Powertrain 86.1 % of 2,041 devices on the 51
   labelled training models against 94.2 % on the held-out ones; size 53.9 % against 53.8 %, both ten points below the internal
   held-out test. The MoreTorque Sonata (3.5 point 4) was the rule, not an exception: the platform split of the lab
   measures the model's transfer to new platforms correctly, and a new device of a known platform is no easier.
3. **Engaged minutes are the powertrain model's weak spot, and the lead group is part of the reason.** 64.8 % window
   accuracy engaged against 92.2 % human-driven; the lead-masked full model 73.3 / 93.2, the speed-only V3 73.5 / 86.0.
   The pedal flags, the one input group no other external set offered, do not help (V2 84.6 % of devices against V3
   92.6). A powertrain model for deployment should be the speed-only or longitudinal variant, or the full recipe with
   the lead group dropped or re-built the way an external pipeline builds it.
4. **Size beyond SMALL does not reach new devices with the full model, and more minutes do not help.** MID 8 of 33 and
   LARGE 6 of 30 models (27.6 / 23.5 % of devices); device accuracy falls with the number of windows per device (60.0 ->
   50.3 %) because the misses are per platform (Odyssey 2 of 96, Sienna 15 of 94, Carnival 0 of 15, Lexus RX 7 of 124
   devices). The speed-only variant is at its internal level (53.7 %, LARGE 14 of 30 models), the full model loses nine
   points; the lateral channels carry the full model's decision (yaw mask) but, steering-derived on a new fleet, add
   nothing over the acceleration texture. The internal minivan / SUV LARGE hits were not robust; pickups are.
5. **Hybrids are read in between.** HEV 32.0 %, PHEV 44.6 %, FCEV 36.5 %, MIXED 19.9 % of windows called BEV against ICE
   17.5 % and BEV 78.7 %; plug-ins and the Hyundai hybrids on the BEV side, the Honda hybrids and the Camry 2018 on the
   ICE side.
6. **What the set cannot settle.** The yaw group is steering-derived everywhere (a measured yaw exists only on Ford), the
   lead is a radar gate or the stock-ACC object, the Honda Civic / Fit / HR-V / Freed did not decode (recovered in 2.13), the torque scale
   is per route batch and 55.0 % of the drives are one minute; the size result therefore bounds the transfer of the
   lateral signal from below and does not say whether a measured yaw would restore it.
7. **The Honda gap is closed (2.13).** The nine platforms were lost to a parser rule (the gearbox message named by an
   unset `transmissionType`), not to the data; with the gearbox type taken from the CAN addresses, 2,132 windows of 230
   devices decode and the full model reads them as SMALL (226 of 230 devices, 5 of 5 models) and ICE (227 of 229, 4 of 4),
   the held-out HR-V 5 of 5 on both tasks; the two CIVIC_2022 misses are among the 9 possible-hybrid devices. The
   seen stratum of the phase-3 scope now holds the Civic and Fit and the gasoline CR-V ports, and the pooled phase-3
   decisions are 2,460 / 2,778 (powertrain) and 2,473 / 4,384 (size). On this one-class fleet the yaw mask collapses the
   size call (14.3 % of the devices) and the lead mask changes nothing; the engaged minutes score lower (73.7 against
   91.1 % size, 80.7 against 95.6 % powertrain window accuracy) without flipping a model.
8. **A 2018 probe (2.14): the decoder works on six-year-older logs, both cars are ICE, and the engaged minutes under an
   unseen longitudinal controller are a coin flip for every model.** Human-driven windows 100.0 % ICE (P(BEV) 0.06),
   engaged ones 50.9 % (0.47) with 2018 openpilot sending the brake / ACC commands itself; the speed-only V3 does not
   escape it (64.3 % of the windows). Size: Civic MID by 0.491 to 0.485 from its engaged minutes, RAV4 SMALL 0.54; V3 both
   SMALL. Two cars, one driver each: recorded as probes.
9. **The steering-derived yaw is biased by understeer, and this concerns every CAN set (2.14).** Against the logged
   `yawRate` of the 2018 logs (2018 openpilot's vehicle-model output, not a sensor; the lab's IMU / pose yaw shows the
   same shape, `UNDERSTEER.md`) the bicycle-model yaw is right at 1-5 m/s (ratio 0.98 / 0.87) and 1.5x too large at 20-25 m/s,
   2x at 30-40 m/s (1 + k v^2, k about 0.001 s^2/m^2; correlation 0.999, same sign). The lateral channels of `ccs`,
   `ccs_honda` and `comma2k19` are therefore inflated on the highway minutes, which sharpens point 6: the size result of
   phase 3 bounds the lateral transfer from below with a known, speed-dependent input bias, and an understeer-corrected
   or measured-yaw re-decode (section 4) is the experiment that separates the input from the model. Done: 3.7.

### 3.7 What the yaw correction changes (2.15)

1. **The lateral size signal transfers in aggregate once the yaw is derived consistently with training; the gain is uneven
   across classes (point 2).** The same 45,763 windows of the
   same 4,061 devices with the bicycle yaw divided by 1 + k v^2 (one global k = 0.00079289 s^2/m^2 fitted on the lab's own
   IMU / pose yaw, 62 platforms, never on the external labels): full-model size device accuracy 54.1 -> 63.2 % (macro-F1 46.4 ->
   60.3) against the internal held-out vehicle accuracy of 63.5 / 59.7; models 63 -> 72 of 113; platforms 95 -> 108 of 161;
   seen 53.9 -> 64.3 % of devices, held-out 53.8 -> 58.0 %, absent unchanged at 88.9 %. On this evidence the "nine points lost" of point 4
   were a speed-dependent bias of the decoded input rather than a model limit; the corrected full model is 9.5 points above the
   speed-only V3 on the same windows, the margin the lateral channels add internally.
2. **LARGE is restored to its internal level, MID is not.** Device recall LARGE 23.5 -> 51.9 % (internal 50.6), MID 27.6 ->
   34.7 (internal 47.3), SMALL 91.1 -> 90.5 (internal 79.2); LARGE models 6 -> 17 of 30, MID 8 -> 8 of 33. Recovered: all four
   Pacifica platforms, Palisade, CX-9, Sienna 4th gen, Rivian R1, Maverick, Ranger, G80, Hyundai Genesis, both Escalades, the
   Crafter (15 LARGE platforms, none lost); the Honda Odyssey / Pilot / Ridgeline, Carnival, Atlas, Explorer, G90 / GV80 and
   the older Sienna platform stay missed, the Pilot and Carnival moving to MID. MID gains the Accord, Ioniq 6, Santa Fe
   Hybrid and a Sorento device and loses the Tesla Model 3 and the Mach-E, two of the BEV platforms that carried the as-run
   MID hits; the Jeep Grand Cherokee (borderline SMALL) is called MID after the correction. The minivan / three-row-SUV
   reading of point 4 ("the internal minivan / SUV LARGE hits were not robust; pickups are") and the "truck-like LARGE" reading
   of 3.2 point 3 no longer hold for the Pacifica, Palisade, CX-9 and Sienna 4th gen and still hold for the Honda ports and the
   Carnival.
3. **Powertrain and the small sets are unchanged in substance.** `ccs` powertrain 87.8 -> 88.9 % of devices, 86 -> 85 of 88
   models (the F-150 Lightning at P(BEV) 0.52 -> 0.46 on 6 devices); the yaw-masked runs of both decodes are identical, the
   speed-only V3 is untouched. `ccs_honda_us` 226 -> 215 of 230 devices SMALL (10 to MID and 2 to LARGE, one back from MID;
   every platform SMALL, powertrain
   99.1 -> 99.6 %); `comma2k19_us` 1 of 2 SMALL and 2 of 2 ICE as before. V1 moves further than the full model in the same
   direction (size 48.2 -> 59.0 %, powertrain 76.4 -> 84.9 % of devices on `ccs`).
4. **Status of the claim.** Post hoc and read next to the as-run results, which stay primary; one global k fitted on the lab's
   data, validated on comma2k19 against a vehicle-model yaw (not a sensor); a residual of about +15 % at 30-40 m/s and a
   geometry error of more than 10 % on a third of the platforms remain in the external features; the pooled totals of 3.1 /
   3.2 exclude the `*_us` sets by design. With the corrected decode as the pipeline going forward, the size transfer of the
   full model on a device-disjoint fleet is at the level of the lab's held-out test, SMALL and LARGE at or above their internal
   recall and MID below it.

## 4. What the owner's downloads would add, and what remains

Status on disk (`<external>/DOWNLOAD_STATUS.md`, 10 Nov): commaSteeringControl (36 archives, 21 GB) and MoreTorque (four
platform folders, 75 GB) consumed and scored in phase 2 (2.10, 2.11); commaCarSegments budget list (49,563 segments, 74 GB)
downloaded, decoded and scored in phase 3 (2.12); comma2k19: the two chunks (Chunk 1 RAV4, Chunk 3 Civic) were completed by the owner with the resumable multi-connection
downloader and decoded and scored the same evening (2.14); the nine Honda platforms of the commaCarSegments budget list were
re-decoded and scored as `ccs_honda` the same day (2.13); CI test routes and CANdid not downloaded. 11 Nov: the three
steering-derived sets re-decoded with the understeer-corrected yaw (`ccs_us` 2.9 h, `ccs_honda_us` 10 min, `comma2k19_us`
3 min, 6 workers), scored with the nine runs each and collected (2.15, 3.7); the understeer fit on the lab's own drive tables
(`understeer_fit.py`, `UNDERSTEER.md`) precedes them.

* **commaCarSegments (done, 2.12):** the full list (`select.txt`, 60,619 segments) is a superset of the budget list that
  was decoded; the 11,056 remaining segments are held-out platforms capped at 1,000 (Corolla, Corolla TSS2, Lexus RX /
  RX TSS2, Sienna, Atlas), which would add devices to models whose decisions are already systematic (Lexus RX 7 of 124
  devices, Sienna 15 of 94, Atlas 0 of 32) and is not expected to change a number. Of the two things that would, the first
  is done: (i) the nine Honda platforms (2,198 segments) were recovered on 10 Nov by the gearbox option of `ccs_decode.py`
  (2.13; the cause was a parser rule, not the DBC set, so no second car stack was needed). Still out of the set: the 5
  Honda segments of 3 devices whose wheel-speed and VSA messages never reach the pt bus, the 24 segments without a
  CarInterface in opendbc 0.3.1 (20 Tucson 4th-gen, 3 CR-V Hybrid 6G, 1 Sonata Hybrid 2024) and, as a label question, the
  9 HONDA_CIVIC_2022 devices whose gearbox message says `automatic` (most likely the Civic Hybrid; kept under the
  platform's ICE label, reported with and without in 2.13, and a candidate for an `unknown` powertrain or a per-device
  split by the owner). (ii) The yaw question is answered for the understeer part by the re-decode below (2.15): with the
  corrected derived yaw the size transfer is at the internal level; what a logged yaw would add beyond that is open (Ford only
  in this set, and the 2018 comma2k19 `yawRate` is a model output).
* **comma2k19 (done, 2.14):** the two cars decode on 2018 logs with four decoder options (layout, fingerprint-built
  `CarParams`, `controlsState` automation flag, waiver for the HUD messages a 2018 log cannot carry) and are ICE at car
  level, the Civic a size coin flip; the set's lasting value is the logged `yawRate` it carries (2018 openpilot's
  vehicle-model output, not a sensor), which exposed the understeer bias of the derived yaw; the re-decode changes nothing
  at car level on these two cars (2.15).
* **Done (11 Nov): the understeer-corrected re-decode of the three steering-derived sets (2.15, 3.7).** The derived yaw
  overestimates the true one as 1 + k v^2 (2.14), so the lateral channels of `ccs`, `ccs_honda` and `comma2k19` were 1.5-2x
  too large on highway windows. Step 1 was done differently from the plan above: k was fitted on the lab's own IMU / pose
  yaw (6,657 drives, 78 eligible platforms; `understeer_fit.py`, `UNDERSTEER.md`), not on comma2k19 or the Ford windows,
  because the comma2k19 `yawRate` turned out to be 2018 openpilot's vehicle-model output and a fit on the external set
  would have tuned the input on the test data; one global k = 0.00079289 s^2/m^2 (hours-weighted median over 62 train /
  validation platforms) went into the new `--yaw derived-understeer --understeer-k K` option of `ccs_decode.py`. Step 2:
  the three re-decodes (2.9 h / 10 min / 3 min with 6 workers; only the three yaw-group channels changed, verified by the
  collector), the nine scoring runs per set, `collect_scores.py` (section "Understeer-corrected yaw re-decodes" of
  `RESULTS_TABLES.md`), `ccs_selftest.py` and section 2.15. What it settled: the full model's size loss on `ccs` was the
  inflated lateral input (54.1 -> 63.2 % of devices, internal 63.5; LARGE 23.5 -> 51.9 % of devices, internal 50.6), not a
  lateral-signature shift between devices; MID (27.6 -> 34.7 %, internal 47.3) is the part that is not the input; the
  powertrain result does not move (87.8 -> 88.9 %).
* **Recommendation for every future external decode.** Make `--yaw derived-understeer --understeer-k 0.00079289` the
  default of `ccs_decode.py` for steering-derived sets (the as-run `--yaw derived` stays available for the record and is
  what 2.12-2.14 describe); prefer a logged yaw where a brand provides a real one (`--yaw measured`: Ford's `CarState.yawRate`
  in `ccs`, after a unit and sign check, which has not been done; the 2018 comma2k19 field is not one); and check the
  logged `carParams` geometry per platform before trusting the derived yaw at any speed, because the two-parameter lab fits
  put the steerRatio / wheelbase / angle scale off by more than 10 % on a third of the platforms (a > 1.1 on 32.1 %, a < 0.9
  on 10.3 %), a bias the understeer term cannot fix: a per-platform scale a estimated on the lab's measured yaw is a
  calibration and must not enter an external feature, but a check of the opendbc values against the manufacturer's
  specifications can. Also keep: drop or back-fill sample 0 of each segment, the all-zero `CarState` of the fresh 0.3.1
  parser (`v` = 0 at k = 0 in every window of the three sets), so that the first grid sample is a decoded one; the lead gate
  could reject lead speeds below zero (oncoming traffic, 2.13 caveats). Neither artefact was touched by the re-decode.
* **Remaining after the correction.** The +15 % residual at 30-40 m/s with one global k (per-platform k1 spans 0.0002-0.0024
  over the lab's train / validation platforms, -0.0002 to 0.0025 over all eligible), the geometry error a above, the Honda LARGE ports (Odyssey, Pilot, Ridgeline) and the Carnival
  that move towards MID but not to LARGE, MID itself (8 of 33 models before and after; the ICE sedans and crossovers are
  still SMALL to the model), and the question whether a sensor yaw would close the residual (CANdid, below, would offer one
  on SMALL cars only).
* Not downloaded, lower value: CI test routes (internal evaluation only, unmasked VINs), CANdid (10 SMALL ICE cars, raw
  CAN incl. steering and yaw: it would offer a measured yaw on SMALL cars, where the size model does not fail), Ford
  Multi-AV (HEV, kinematics only).
* **Model-side steps suggested by the three phases**, in the order of expected value: (a) a powertrain model without
  the lateral channels and without the lead group, or with the lead group in the dropout augmentation and re-built the
  way an external pipeline builds it (radar gate / stock-ACC object), since on commaCarSegments the lead-masked full
  model (91.3 % of devices) and the speed-only V3 (92.6 %) beat the full model (87.8 %) and the loss is on engaged
  minutes; a training-time re-weighting of engaged windows for the powertrain task is the cheaper alternative; (b) a
  size model trained with steering-derived yaw / curvature / lateral acceleration (the understeer-corrected bicycle model
  on the lab's own steering angle, 2.15) in place of or alongside the IMU / pose yaw, so that inference on an external fleet
  matches training even where the geometry is off,
  plus device-level augmentation; the MIL episode models of Part II remain the candidate for MID, which no set and no
  stratum recognises for ICE sedans and crossovers; (c) hybrids as a third powertrain class or an ordinal electrification
  target, since the model already places them between ICE and BEV (HEV 32.0 %, PHEV 44.6 % of windows BEV) and the
  commaCarSegments set gives 1,093 HEV and 214 PHEV devices to train or calibrate on; (d) the two steps of phase 1
  remain for the GNSS / OBD sets: a V2 / V3 variant trained with `a := smooth_derivative(v)` as an augmentation, and a
  V1-style variant with `steer_pressed` in the dropout groups.

## 5. Files and reproduction

* Scores and probabilities per set: `<external>/<set>/work/score__<checkpoint set>__<mask>.md/.json` and
  `probs__<checkpoint>__<mask>.npz` / `probs__<set>__ens__<mask>.npz` (56 score files: `full` = variant checkpoints or the
  full model as is, `mask-V1/V2/V3` = full model with that mask).
* `experiments/transfer/run_scoring.sh` (the 28 scoring commands, nice -n 10, GPU; 2.5 min in total),
  `collect_scores.py` -> `RESULTS_TABLES.md` + `results_summary.json` (every table of this report incl. the per-campaign,
  DAS-split, VED weight-bin and pooled-decision tables), `internal_reference.py` -> `internal_reference.md/.json`
  (seed-ensemble references incl. the full model under the V1 / V2 / V3 masks; masked probabilities cached under
  `<external>/_internal_ref/`), `internal_per_model.py` -> `internal_per_model.md`, `diag_derived_accel.py` ->
  `diag_derived_accel.md/.json`.
* `experiments/transfer/review_recompute.py` (reviewer check, 9 Nov; phase 2 and phase 3 added 10 Nov): recomputes the headline numbers of
  sections 2.1, 2.2, 2.5-2.7, 3.1 and 3.2 (phase 1, sections 1-9 of its output) and of sections 0, 2.10, 2.11, 3.1, 3.2 (phase 2,
  sections 10-19: every level table, the per-archive / per-car / per-stratum decisions, the mask and `--das any` sensitivities,
  the pooled totals) directly from the `probs__*__ens__*.npz` files and the metas, independent of `score.py` /
  `collect_scores.py`; it also re-checks the phase-2 labels (strata against `build/index/drives.parquet`, footprints against the
  lab table, size classes against the cuts, hybrid-free archive names), the sign conventions (steering angle vs yaw on both
  sets, lead relative speed = lead minus ego on MoreTorque, raw `latAccelLocalizer` right-positive in the zips with
  `--raw-sign-check`), the 598-599-row padding rule on the window tensor and the identifier hygiene of the phase-2 outputs
  (`--phase 2`; log `<external>/logs/review_recompute_phase2.log`). Every recomputed figure equals the one quoted here (the
  review corrected two roundings in the `--das any` sentence of 2.11).
* Log: `<external>/logs/scoring_main.log`. Nothing under `build/`, `release/`, `private/` or the raw library was touched;
  `experiments/RESULTS.md` (the internal results) is unchanged.
* Phase 2 (9 Nov, evening): `experiments/transfer/run_scoring.sh csc moretorque` (the `csc` and `moretorque` cases: 3 + 7
  scoring commands, nice -n 10, GPU shared with another job; 4 min), plus the sensitivity runs of 2.11
  (`score.py --dataset moretorque --mask-channels lead | yaw | torque,pedals` and
  `score.py --dataset moretorque_dasany` on the `ds_moretorque.py --stage assemble --das any --dataset moretorque_dasany`
  re-cut, whose work folder links to the decoded 10 Hz tables of `moretorque`); log
  `<external>/logs/scoring_csc_moretorque.log`, re-cut log `<external>/logs/moretorque_dasany_assemble.log`. Scores:
  `<external>/csc/work/score__transfer__{size__gru,bev_ice__resnet}__{V1,V3}__full.*`,
  `score__split_platform__*__mask-V1.*`; `<external>/moretorque/work/score__*` (full, V1 / V2 / V3 variants, mask-V1 /
  V2 / V3, mask-lead, mask-yaw, mask-3ch-b54e96 = torque + pedals); `<external>/moretorque_dasany/work/score__*`.
  `collect_scores.py` gained the two sets (`DATASETS`, `RUNS["V1"]`, `RUNS["V0mt"]`), the per-archive / per-stratum
  tables of commaSteeringControl (`csc_tables`) and the per-vehicle / automation-state / speed-regime tables of
  MoreTorque (`moretorque_tables`); the phase-1 pooled-decision scopes are unchanged (`PHASE1`) and the new sets appear
  as extra scopes. The per-archive and per-car tables of 2.10 / 2.11 were generated from the `probs__*__ens__*.npz`
  files and the metas directly (same rule as `collect_scores.py`), the level tables are copied from the `score__*.md`
  files. Nothing under `build/`, `release/`, `private/`, `work/` or the raw library was touched; no device id, route id
  or absolute path appears in the phase-2 outputs (pseudonymous `veh_` / `drv_` ids and public platform names only).
* Phase 3 (10 Nov): download by the owner (`download_owner_sets.sh --only ccs` with the budget list `select_budget.txt`,
  49,563 segments, 74 GB; logs `<external>/logs/download_*.log`); decode `ccs_decode.py --segments-root <external>/ccs/segments
  --list <external>/ccs/select_budget.txt --dataset ccs --workers 6` (3.2 h; outputs `<external>/ccs/work/windows.npz`,
  `meta.parquet`, `windows_summary.json`, `segments_qc.csv` (one row per decoded segment: `canValid`, lead source, radar
  availability, yaw source, torque, automation share), `decode_summary.json` (accounting and skip reasons)); scoring
  `run_scoring.sh ccs` = 9 commands (nice -n 10, GPU; log `<external>/logs/scoring_ccs_*.log`): the full models
  (`score__split_platform__{size__gru__S9,bev_ice__resnet__B5}__full`), the V1 / V2 / V3 variants
  (`score__transfer__{size__gru,bev_ice__resnet}__V{1,2,3}__full`), the full models with the V1 / V2 / V3 presets masked
  (`__mask-V{1,2,3}`), the yaw-group mask (`__mask-yaw`) and the lead-group mask (`__mask-lead`), each as `.md` / `.json`
  with the `probs__*__ens__*.npz` ensembles beside them; `collect_scores.py` (log `<external>/logs/collect_ccs.log`)
  renders the eight commaCarSegments tables of `RESULTS_TABLES.md` (`ccs_tables`: decisions per platform, pooled per
  stratum with 300-resample intervals, held-out models against `internal_per_model.md`, seen and absent models, lead
  source and automation state, evidence per device, predicted-share confusion, unlabelled powertrains) and adds the set
  to the pooled-decision scopes (`PHASE3`); `ccs_selftest.py` tiles the four `ccs_sample` windows into 14 synthetic
  platforms under `<external>/_selftest_ccs`, runs the nine commands and checks the tables (strata sum to `all`, units right
  within support, pooled rows equal the score files, no long hex token). The automation-state split of the other
  powertrain runs quoted in 2.12 (e) was computed from the `probs__*__ens__*.npz` files and the meta with the same rule as
  `collect_scores.py`. `review_recompute.py --phase 3` (sections 20-33, added by the phase-3 review on 10 Nov; log
  `<external>/logs/review_recompute_phase3.log`) recomputes every table of 2.12 cell by cell and the figures of its Reading
  bullets from the `probs__*__ens__*.npz` files and the meta, the decode accounting against `decode_summary.json` /
  `segments_qc.csv` / `select_summary.json`, the labels and strata (footprints against both spec tables, strata against
  the drives index, powertrain against the lab table, the lab-device exclusion by intersection count), the internal
  references, the window-tensor identities and the identifier hygiene. The review corrected in this file: the 1,271 no-object
  windows split (928 stock-ACC + 343 radar platforms), the public-spec platform count (22; 13 of the 14 absent models), the
  multi-platform device counts (88 under two, 3 under three), the brake / steering shares (13 / 22 %), three truncated
  engaged-minute device accuracies (84.3 / 73.9 / 81.8), the held-out size agreement count (19 internal hits, 19 agreements),
  the Kona device split, the HEV / MIXED device counts (1,093 / 275 unique devices), the V3-mask model count (1 of 63), the
  yaw-mask reading (69.6 % of the devices called LARGE; 88.3 % is the LARGE recall), the no-borderline interval at 300
  resamples and the pooled totals under the device's own label. Nothing under `build/`,
  `release/`, `private/` or the raw library was touched; the device and route ids of commaCarSegments exist only under
  `<external>/ccs/segments` and in the selection lists, never in the repository or the work folder (pseudonymous
  `ccs_v_` / `ccs_d_` ids and public platform names only).
* Phase 3, addenda (10 Nov, evening): `ccs_decode.py` gained four options that are off by default and leave the `ccs` path
  byte-identical (checked by sha1 of the window tensor and meta on the check segments): `--honda-gearbox-from-can`
  (Honda only; the `gearbox=` note and the `gearbox` column of `segments_qc.csv`), `--layout comma2k19`,
  `--fingerprint-map <folder>=<PLATFORM>,...` (`cp_source` / `cp_logged_diff` columns, `cp=` note), `--das-from
  controlsState` (`das_source` / `controls_enabled_frac` columns) and `--ignore-absent-messages` (`absent_msgs` column);
  `ccs_honda_list.py` (the nine platforms' segments within the budget list, and the 20-segment check list; the lists stay
  on the data disk) and `ccs_honda_check.py` (option off / on on the 20 check segments, and the option against the 2024-08
  lab car stack on the 21 probe segments of the diagnosis; tables in `CCS_PIPELINE.md` "Honda re-decode"). Decodes:
  `ccs_decode.py --list <external>/ccs/select_honda.txt --dataset ccs_honda --workers 6 --honda-gearbox-from-can` (467 s;
  log `<external>/logs/ccs_decode_honda.log`) and `ccs_decode.py --segments-root <external>/comma2k19/segments --layout
  comma2k19 --fingerprint-map Chunk_1=TOYOTA_RAV4,Chunk_3=HONDA_CIVIC --das-from controlsState --ignore-absent-messages
  --dataset comma2k19 --workers 6` (153 s; log `<external>/logs/ccs_decode_comma2k19.log`); work folders
  `<external>/ccs_honda/work` and `<external>/comma2k19/work` (`windows.npz`, `meta.parquet`, `windows_summary.json`,
  `segments_qc.csv`, `decode_summary.json`). Scoring: `run_scoring.sh ccs_honda` and `run_scoring.sh comma2k19`, the nine
  runs of the `ccs` case each (full, V1 / V2 / V3, mask-V1 / V2 / V3, mask-yaw, mask-lead; logs
  `<external>/logs/scoring_ccs_honda_wrapper.log`, `scoring_comma2k19_wrapper.log`); `collect_scores.py` (log
  `<external>/logs/collect_ccs_honda.log`) registers `ccs_honda` as a second phase-3 set (its own `ccs_tables` section
  and the `PHASE3` pooled scope summed over both sets) and `comma2k19` as a scope of its own (`OLD_SOFTWARE`, per-dataset
  block and automation rows only); `ccs_selftest.py` passes. The per-device figures of 2.13 (the possible-hybrid split,
  the two device-level misses, the lead and automation shares) and of 2.14 (the per-car automation split for size, the
  V3 per-car probabilities, the route-folder sizes and gaps) were computed from the `probs__*__ens__*.npz` files, the metas,
  `segments_qc.csv` and the segment folders with the same rules as `collect_scores.py`; the derived-yaw ratio table and the
  checks against the logged 2018 `carState` of 2.14 come from the comma2k19 verification of `CCS_PIPELINE.md` (lab
  `LogReader`), the sample-0 and oncoming-lead artefacts from the window tensors. Review coverage: `review_recompute.py --phase 3`
  sections 34-41 (phase-3b review, 10 Nov, evening; same log) recompute every table of 2.13 (a)-(e) and the results table of
  2.14 cell by cell, the figures of their Reading bullets and caveats, the decode accounting of both sets against
  `decode_summary.json` / `segments_qc.csv` (and the budget list), the pooled totals of sections 0 / 3.1 / 3.2, the labels and
  strata of both sets, the window tensors (the sample-0 and oncoming-lead artefacts) and the identifier hygiene of both work
  folders; `comma2k19_yaw_check.py` (lab venv, `$PY`) re-derives the derived-yaw ratios of 2.14 from the logged
  `carState.yawRate` (`yaw_check.json` on 5 segments per car, `yaw_check_all.json` on all 381 windows, in the comma2k19 work
  folder; logs `<external>/logs/comma2k19_yaw_check*.log`). The review corrected in this file: the window-set speed and pedal
  shares of `ccs_honda` (18.2 m/s, 37 / 15 / 18 %; the earlier 17.7 / 36 / 16 were means over all decoded segments) and of
  comma2k19 (22.8 / 20.9 m/s; brake 0.12 / 0.12 in `CCS_PIPELINE.md`), the lead attribution of 2.13 (909 radar-platform windows,
  887 with an object; 1,223 on the radarless ports), the multi-platform device count (one of the nine possible hybrids carries
  windows under three platforms; the other multi-platform device is a `cvt` device), the double-counted seen devices (123 + 96 +
  7 = 226 for 224 devices), the P(BEV) reading of the pooled possible-hybrid device (0.485 is P(BEV)) and the `cruise_frac` /
  `controls_enabled_frac` wording of 2.14. Nothing under `build/`, `release/`, `private/` or the raw library was touched; the
  comma2k19 device and route folder names exist only under `<external>/comma2k19/segments`, never in the repository or the
  work folder.
* Understeer-corrected re-decode (11 Nov): `experiments/transfer/understeer_fit.py` (the fit of k on the lab's released drive
  tables: per-drive filters and angle offset, per-platform 1-par / 2-par / speed-bin fits with one robust pass, the global
  hours-weighted median over train / validation platforms, the comma2k19 validation against the logged 2018 `yawRate`;
  `nice -n 10 $CPY ... --workers 6 --json experiments/transfer/understeer_fit.json`, scratch under `<data-root>/tmp/understeer/`),
  `UNDERSTEER.md` (method, spread, why one global k, the geometry scale a, k against footprint and class, the 78-platform table,
  the comma2k19 validation, caveats incl. the vehicle-model nature of the 2018 `yawRate`), `understeer_fit.json` (numbers only;
  platform names, no device / route identifiers, no machine paths); `ccs_decode.py --yaw derived-understeer --understeer-k K`
  (`CCS_PIPELINE.md` "Understeer-corrected yaw": formula, bookkeeping in `segments_qc.csv` / `meta.note` / `decode_summary.json`,
  the byte-identical default path, the smoke on comma2k19 and the three commands). Decodes: `<external>/ccs_us/work`,
  `<external>/ccs_honda_us/work`, `<external>/comma2k19_us/work` (`windows.npz`, `meta.parquet`, `windows_summary.json`,
  `segments_qc.csv`, `decode_summary.json`; logs `<external>/logs/ccs_decode_us.log`, `ccs_decode_honda_us.log`,
  `ccs_decode_comma2k19_us.log`). Scoring: `run_scoring.sh ccs_us ccs_honda_us comma2k19_us`, the nine runs of the `ccs` case per
  set = 27 runs (full, V1 / V2 / V3, mask-V1 / V2 / V3, mask-yaw, mask-lead; logs `<external>/logs/scoring_us_small_*.log`,
  `scoring_ccs_us_*.log`). `collect_scores.py` (`yaw_correction_tables`; log `<external>/logs/collect_us.log`) registers the three
  sets with the kind `ccs_us`, keeps them out of every pre-existing table and renders the section "Understeer-corrected yaw
  re-decodes" at the end of `RESULTS_TABLES.md` (alignment check, per-stratum rows for the full model and V1, platform decision
  flips, devices right per class, the yaw-masked reference, one summary line per set) with the numbers in
  `results_summary.json["yaw_correction"]`; `ccs_selftest.py` builds the synthetic `_selftest_ccs_us` pair, checks the alignment
  result, the read-back k, the delta arithmetic, the strata sums, the flip bookkeeping and the masked-run identity, verifies that
  two misaligned copies are refused and that the collector's output before the understeer section is byte-identical with and
  without the pair (PASS, log `<external>/logs/ccs_selftest_us.log`). `experiments/transfer/us_heldout_compare.py` ->
  `us_heldout_compare.md` (table (d) of 2.15: the held-out models of `ccs` against `ccs_us` from the `probs__*__ens__full.npz`
  files and the metas, same rules as the collector). The per-class device confusion, the automation-state split, the
  speed-binned window recalls, the per-platform device counts of the LARGE / MID platforms, the comma2k19 per-car probabilities
  and the Honda per-platform counts quoted in 2.15 were computed from the same probability files and metas with the same rules
  (the mean window speed from `windows.npz` with sample 0 excluded). Review coverage: `review_recompute.py` sections 42-50 (phase 3c,
  same log `<external>/logs/review_recompute_phase3.log`) recompute from the probability files and metas of both members of each pair
  every table of 2.15 ((a)-(e); table (d) independently of `us_heldout_compare.py`), the figures of its Reading bullets, the understeer
  block of 0, 3.7, the new parts of 4, the pointers in 2.12-2.14 and the result paragraph of `CCS_PIPELINE.md`; verify the alignment of
  the pairs on the window tensors including the 1 / (1 + k v^2) factor itself; spot-check `UNDERSTEER.md` against `understeer_fit.json`,
  the scratch tables and the decode summaries; and scan the new files and the three work folders for identifiers, paths, dates and the
  acronym. The review corrected: the Honda device movement (10 devices SMALL -> MID, 2 SMALL -> LARGE and one MID -> SMALL, not
  "11 to MID"; 0, 2.13, 2.15, 3.7, `CCS_PIPELINE.md`), the comma2k19 engaged-window accuracies (28.6 -> 73.3 and 24.0 -> 17.4 %,
  the 55.7 -> 74.6 / 48.0 -> 39.8 quoted were the cars' all-window values), the SMALL / MID per-bin deltas (+0.6, -2.4: raw
  differences, as every other delta of the report), the lab k1 range (0.0002-0.0024 over the train / validation platforms, -0.0002 to
  0.0025 over all eligible) and a range (0.79-1.24), the CR-V 5G k1 (0.00111), and in `UNDERSTEER.md` the RAV4 residual on all
  segments (+15.3 %) and its own 1-par k on comma2k19 (0.00106); three sentences were softened to "on this evidence ... rather than
  a model limit" and the aggregate / per-class qualification was added to 3.7 point 1. Nothing under `build/`,
  `release/`, `private/` or the raw library was touched; the re-decodes carry the same pseudonyms as their originals, no device
  or route id appears in the repository.
