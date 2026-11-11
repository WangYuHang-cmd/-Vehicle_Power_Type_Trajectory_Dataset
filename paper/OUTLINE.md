# Paper outline and single source of numbers (binding for every section writer)

Target: IEEE Intelligent Vehicles Symposium (IV) paper (IEEEtran, two columns, 6 pages, at most 8 with extra-page charges, double-blind review). Working
title: **"Reading Vehicle Size and Powertrain from Motion: A Trajectory–CAN Benchmark with Held-Out Car Models,
Pre-Registered Methods and Label Ceilings"** (writers may propose a shorter alternative in a LaTeX comment).

Author block: leave the template's commented author block as a TODO (the user fills names / affiliations).

## Message in one paragraph
From 60-s windows of trajectory and CAN signals (speed, acceleration, steering angle / rate / normalised torque, yaw
rate, curvature, lateral acceleration, pedal flags, automation flag, lead-vehicle distance / relative speed / speed /
acceleration / headway / time-to-collision) we ask whether a vehicle's size class and powertrain type can be inferred for
car models never seen in training. We release a dataset (9,062 drives, about 3,950 h, 1,431 pseudonymous vehicles, 174
car models) with model-disjoint and device-disjoint splits and window / drive / vehicle protocols; benchmark four
sequence architectures (1-D ResNet, bi-GRU, Transformer, bidirectional Mamba-2), feature baselines and LLM-proposed
physical features; then run a pre-registered method campaign (model-disjoint internal folds, training-recipe changes,
an episode-centric multiple-instance model). Powertrain (BEV vs ICE) reaches 92.6–93.0 drive-level macro-F1 on unseen
car models (vehicle level 92.8–93.4); size (three footprint classes) improves from 62.9 to 67.1 at drive level but
stays far from 90, and zero-training oracle analyses show the gap is set by the label resolution and the composition of
the training fleet, not by model capacity or evidence per drive.

## Contributions (use these, in this order)
1. A public, privacy-preserving trajectory + CAN dataset for vehicle-type inference with fixed labels (footprint size
   classes, powertrain), two split schemes (held-out car models = headline; held-out devices) and window-, drive- and
   vehicle-level protocols with cluster-bootstrap confidence intervals over car models.
2. A benchmark of four sequence architectures, hand-crafted statistics and LLM-proposed physical features under those
   protocols, showing that architecture is not the bottleneck (all deep models within about 2 points) and that drive-level
   aggregation adds 7–12 points.
3. A pre-registered method study with model-disjoint internal folds: car-model-balanced sampling, auxiliary physical
   targets (footprint regression, ordinal size, curb mass, four-class powertrain), physics-safe augmentation and DAS
   down-weighting raise the drive-level result by 4–6 points (size) and 2–4 points (BEV vs ICE); an episode-centric
   multiple-instance model over acceleration, deceleration, stop/launch, car-following and steering transients matches but
   does not beat the window recipes on the test models.
4. Zero-training ceiling analyses (per-car-model majority oracle, footprint-noise oracle, curb-mass oracle, evidence
   sufficiency, separability of held-out models) that quantify why size cannot reach 90 on unseen car models under a
   footprint cut, and what would make it reachable.

## Section plan, files and length budget (words of prose, excluding tables)
| file | section | words | owner |
|---|---|---|---|
| sections/00_abstract.tex | Abstract (one paragraph, 170–200 words, numbers: drives, hours, models; BEV vs ICE 92.6–93.0 drive-level; size 67.1 vs ceiling) | 190 | W1 |
| sections/01_intro.tex | I. Introduction: motivation (vehicle-type-aware traffic / simulation / infrastructure analytics from kinematics only; privacy), problem statement (two tasks, unseen car models), why it is hard (continuous physics vs hard labels, DAS periods, fleet composition), contributions list | 600 | W1 |
| sections/02_related.tex | II. Related work: (a) driving-behaviour differences between electric and combustion vehicles (keep general, cite only keys below; the user adds their own prior work: leave `% TODO cite own EV-vs-ICE paper`), (b) vehicle / driver identification from kinematics and CAN, (c) sequence models for time series (ResNet, GRU, Transformer, Mamba), (d) shortcut learning and group-disjoint evaluation, (e) LLMs for time series and for feature / program discovery | 450 | W1 |
| sections/03_dataset.tex | III. Dataset and protocols: source and extraction (open driving-automation logs, 10 Hz car-state stream, sample-indexed tables), channels (20 official inputs; list them), labels (footprint = wheelbase x mean track; thresholds 4.43 / 4.88 m^2 locked before any model; powertrain), statistics table (Table I), splits (held-out car models; validation shares models with train, only test is model-disjoint; identity groups; held-out devices), protocols (window test_full / test_balanced / test_human_only; drive_full; drive_human_only; vehicle_full), metric and CI, privacy (pseudonyms, no location / time / video), availability (HF + GitHub) | 900 | W2 |
| sections/04_methods.tex | IV. Methods: A. window models (ResNet1D on 10 Hz; conv stem stride 4 -> 150 tokens for bi-GRU / Transformer / Mamba-2; Mamba-2 = pure-PyTorch SSD chunked scan, bidirectional); B. training recipe (class-balanced sampling (N/(K n_k))^0.5 capped 10, AdamW, warm-up + cosine, label smoothing 0.05, early stopping); C. model-disjoint internal folds and pre-registration (3 folds over train+validation pool, fleet rule, fixed epochs = median of fold best epochs, test once); D. recipe changes S9 / B5 (car-model-balanced sampling hours^0.5; heads: footprint log-regression, CORAL ordinal, log curb mass, powertrain-4 with hybrid windows; augmentation: lateral sign flip of six channels, 3–15 s masking with lead_valid = 0, channel-group dropout, noise 0.02; DAS down-weighting beta 0.5; group-DRO-lite); E. episode MIL (8 episode types, onset-anchored 18-s crops, speed-indexed a(v) grid, encoder d = 128, gated attention, window-embedding instances, auxiliary heads); F. LLM-assisted physical features (agents propose code from physics lenses, verifiers, auditor; features -> gradient boosting); G. aggregation to drive and vehicle level (mean of window probabilities = headline; DAS/episode-weighted log-odds = supplementary) | 1000 | W3 |
| sections/05_experiments.tex | V. Experiments: A. set-up (hardware, seeds); B. main results Table II (split_platform; rows below); C. held-out devices Table III (compact); D. folds and recipe sweep (Table IV compact: S0/S2/S5/S9, B0/B4/B5, MIL); E. ceilings (Fig. 2 footprint-noise oracle; majority oracle; mass oracle; separability) and evidence sufficiency (Fig. 1); F. LLM features (Table V compact) and importance; G. aggregation and ensembles; H. per-model errors (which held-out models fail) | 1200 | W4 |
| sections/06_discussion.tex | VI. Discussion and limitations: label resolution vs sensing; LARGE learned as truck-like; DAS controller signature (human-only protocol); shared-model validation lesson; what 90 needs (seen models, vehicle level; label margins > resolution); privacy / ethics; limitations (one logger ecosystem, region skew, label noise, MIXED powertrains excluded) | 500 | W5 |
| sections/07_conclusion.tex | VII. Conclusion (120–160 words) | 150 | W5 |

## Fixed numbers (verified against RESULTS.md, CAMPAIGN.md, ORACLES.md, AGGREGATION.md, RECIPE.md, INTERNAL_FOLDS.md, LLM_FEATURES.md, docs/SIZE_CLASSES.md, docs/PROTOCOLS.md, the dataset card, build/qc/AUDITS.md and paper/tables/ci.json on 7 Nov 2026; copy exactly; percent values are macro-F1 unless stated)

### Dataset (release v1)
- drives 9,062; hours 3,949.8 nominal (3,078.1 moving); pseudonymous vehicles 1,431 (device x platform); identity groups 1,157;
  vehicle platforms 360; brands 38; car models 174; samples with DAS engaged 47.9 %, with any automation activity 62.6 %.
- 60-s windows: 241,476 listed; 201,390 eligible (600 consecutive samples, no gap, >= 20 % of samples moving); 37,942 human-only
  (strict); eligible per task: size 201,169, BEV vs ICE 144,534, four-class powertrain 178,827. Ready-to-train shards hold 194,949
  windows in 41 files; drive tables 8.1 GB, window shards 4.4 GB.
- class support (eligible windows / hours / vehicles / platforms): SMALL 91,369 / 1,522.8 / 673 / 169; MID 56,059 / 934.3 / 392 / 104;
  LARGE 53,741 / 895.7 / 295 / 76; ICE 79,566 / 1,326.1 / 598 / 200; HEV 23,708 / 395.1 / 197 / 57; PHEV 10,585 / 176.4 / 68 / 21;
  BEV 64,968 / 1,082.8 / 366 / 71; powertrain null (MIXED / unknown platforms) 22,563 / 376.1 / 156 / 34 (11.1 % of the drives).
- channels: 10 Hz; 44 columns per drive table; 20 official input channels: v_mps, a_mps2, steer_angle_deg, steer_rate_degps,
  steer_torque_norm, yaw_rate_radps, curv_1pm, lat_acc_mps2, gas_pressed, brake_pressed, steer_pressed, standstill, das_enabled,
  lead_valid, lead_d_m, lead_v_rel_mps, lead_speed_mps, lead_a_slope_mps2, thw_s, ttc_s. Tables are indexed by car-state sample
  (row k = k-th 10 Hz sample, nominal t = k / 10 s, gaps flagged); radar / pose / controller topics are zero-order held causally
  onto the sample times. Pedal magnitudes are missing on many vehicles (gas on 63.9 % of drives, brake on 49.4 %); the pressed
  flags are always valid; the lead vehicle comes from a camera model rather than a radar on 56.2 % of drives; steering torque is
  normalised per drive by its 95th percentile.
- labels: footprint = wheelbase x (front track + rear track) / 2 per platform; SMALL < 4.43 m^2 <= MID < 4.88 m^2 <= LARGE; thresholds
  fixed from the label inventory alone before any classifier (rule: every size x {BEV, ICE} cell >= 100 h, >= 3 car models, >= 5
  identity groups; among admissible pairs maximise the smallest cell (309 h); place the cuts in gaps between platform footprints:
  Kia Sportage 4.421 | Acura TLX 4.439, and Ioniq 5 facelift 4.875 | Hyundai Genesis 4.896). Hours per class: SMALL 1,784 (44.7 %),
  MID 1,124 (28.1 %), LARGE 1,037 (25.9 %), unknown 51. Hours / car models / groups per size x powertrain: SMALL ICE 749 / 55 / 265,
  SMALL BEV 309 / 15 / 99; MID ICE 371 / 42 / 136, MID BEV 494 / 10 / 146; LARGE ICE 457 / 28 / 151, LARGE BEV 460 / 8 / 67.
  Reading: SMALL = compact cars and compact crossovers (Civic, Corolla, Prius, RAV4, CR-V, Bolt, Niro); MID = mid-size sedans and
  two-row SUVs (Camry, Accord, Sonata, Model 3, Model Y, EV6, Ioniq 5 / 6, Mach-E); LARGE = full-size sedans, three-row SUVs,
  minivans, pickups (Palisade, Pilot, Model S / X, Rivian R1, EV9, Sienna, F-150, RAM). Powertrain classes ICE / HEV / PHEV / BEV;
  main task BEV vs ICE; platforms sold with several powertrains that the logs cannot separate get a null label.
- splits: identity group (same device, same vehicle identity or shared drive names) is the unit of both schemes. split_platform:
  41 car models held out as the test split (about 15 % of the hours of each cell; brands contribute both powertrain classes where
  possible); drives of a group that also drove a held-out model are excluded from every split (357 drives in the released index; 394 in the build index before the
  unknown-model drives were dropped); the validation split is carved group-disjoint from the remaining pool and shares 53 of
  its 56 car models with train (54 of 57 in the build index), so only the test split is model-disjoint. split_device: groups assigned at random inside each size x powertrain cell with 70 / 15 / 15 % of the hours; car
  models shared. Lab-owned devices always in train. Eligible windows / hours / groups / platforms: size split_platform train
  143,400 / 2,390.0 / 672 / 216, validation 25,188 / 419.8 / 137 / 85, test 26,144 / 435.7 / 297 / 102; BEV vs ICE split_platform
  test 18,295 / 304.9 / 222 / 77; size split_device test 29,214 / 486.9 / 191 / 120; BEV vs ICE split_device test 21,450 / 357.5 / 131 / 84.
- unit populations of the test split: split_platform size 1,141 drives (85 labelled test drives have no eligible window), 329
  vehicles, 41 car models; human-only windows cover 732 of the drives; BEV vs ICE 836 drives, 244 vehicles, 39 car models, 564
  drives with human-only windows. split_device: size 1,218 drives / 219 vehicles / 81 car models; BEV vs ICE 922 / 143 / 63.
  About 45 % of test vehicles have a single drive (size 46.5 %, BEV vs ICE 44.7 %).
- protocols: window-level test_full / test_balanced (published class-balanced list with group and platform caps) / test_human_only
  (no automation activity in the window nor 10 s before); drive_full = mean of the per-window probability vectors of a drive's
  test_full windows -> arg-max (ties -> earlier class); drive_human_only likewise on human-only windows; vehicle_full = all test
  drives of a pseudonymous vehicle pooled, every window the same weight; vehicle_full_multi = vehicles with >= 2 drives. Metric:
  macro-F1 over the classes present; 95 % interval by cluster bootstrap over car models (1,000 resamples, seed 0; a resample draws
  car models with replacement and keeps every unit of a drawn model once per draw). Chance level about 33 (size) and 50 (BEV vs ICE).
- shortcut audits (group-disjoint, simple classifiers): nuisance metadata (device type, software era, region, pose source, dead
  channels) predict powertrain at 0.32 macro-F1 (chance 0.25) and size at 0.40 (chance 0.33); window statistics of the official
  inputs predict brand at 0.02 (chance 0.03, 38 brands) and platform at 0.00; the same statistics give powertrain 0.31 and size 0.34.

### First round (reference models; split_platform test; mean +- std over 3 seeds; window test_full / drive_full)
- size: 1-D ResNet 53.6+-1.0 / 61.1+-1.3; bi-GRU 54.7+-0.3 / 62.9+-0.8; Transformer 52.5+-0.8 / 61.8+-0.9; Mamba-2 53.1+-0.4 / 61.2+-0.3;
  statistics + logistic regression 43.6 / 48.4; statistics + gradient boosting 49.7 / 54.7; LLM features + gradient boosting
  40.5 / 63.2; LLM features + statistics + gradient boosting 54.1 / 64.9. Window human-only: ResNet 48.0, GRU 50.4, Transformer 47.7,
  Mamba-2 49.8. Drive human-only: ResNet 60.3, GRU 63.3, Transformer 61.3, Mamba-2 61.1. Best single GRU run per-class recall on
  windows: SMALL 80.9, MID 50.9, LARGE 32.4.
- BEV vs ICE: 1-D ResNet 78.4+-0.4 / 91.1+-0.8; bi-GRU 79.2+-0.8 / 89.0+-0.1; Transformer 77.0+-0.4 / 88.4+-0.6; Mamba-2 77.3+-0.7 / 89.4+-0.3;
  statistics + logistic regression 67.2 / 75.9; statistics + gradient boosting 69.5 / 73.2; LLM features + logistic regression
  67.1 / 87.5; LLM features + gradient boosting 69.6 / 89.4 (window human-only 84.2); LLM + statistics + gradient boosting 73.2 / 80.9.
  Window human-only: ResNet 83.2, GRU 85.2, Transformer 83.5, Mamba-2 84.0. Drive human-only: ResNet 89.5, GRU 90.8, Transformer
  90.1, Mamba-2 90.9.
- windows without any automation activity vs windows with automation activity (one run each): size (bi-GRU) 50.2 vs 56.4 macro-F1
  (accuracy 53.9 vs 66.6 %); BEV vs ICE (ResNet B5) 84.0 vs 79.8 (accuracy 87.1 vs 81.2 %).
- cluster-bootstrap 95 % intervals, drive_full (mean of the seeds' bounds): GRU size 62.9 [48.9, 72.4]; ResNet size 61.1 [46.9, 70.8];
  ResNet BEV vs ICE 91.1 [83.4, 95.7]; GRU BEV vs ICE 89.0 [81.8, 93.7].
- vehicle_full of the reference models: size 55.2-57.2 (lower than drive level); BEV vs ICE 88.6-90.6 (same as drive level).
- held-out devices (split_device, first round, window / drive): size ResNet 68.4+-3.0 / 77.4+-2.2, GRU 70.2+-0.9 / 78.1+-1.4, Transformer
  68.9+-0.4 / 78.0+-0.2, Mamba-2 (1 seed) 69.8 / 77.8, statistics + GBM 70.8 / 76.8, LLM + statistics + GBM 72.4 / 80.6; BEV vs ICE ResNet
  82.4+-0.7 / 96.2+-0.5, GRU 87.4+-0.4 / 96.5+-0.1, Transformer 86.1+-0.4 / 95.2+-0.7, Mamba-2 (1 seed) 87.0 / 96.7, statistics + GBM
  82.7 / 90.7, LLM + GBM 74.8 / 95.4, LLM + statistics + GBM 84.4 / 94.1.
- model sizes: 1-D ResNet 2.31 M parameters, bi-GRU 0.73 M, Transformer 0.70 M, Mamba-2 1.07 M; GRU / ResNet / Transformer 3-10 min per
  run, Mamba-2 45-65 min (one RTX 5080, 16 GB).
- training set-up of the reference recipe: 20 standardised channels (training-split statistics, clipped, missing -> 0), 2-layer
  convolutional stem (stride 4 -> 150 tokens of width 128) for GRU / Transformer / Mamba-2 (4 layers Transformer and Mamba-2, 2
  bidirectional GRU layers), ResNet on the 10 Hz signal; class-balanced sampling with weights (N / (K n_k))^0.5 capped at 10; AdamW,
  warm-up + cosine; label smoothing 0.05; batch 256; at most 25 epochs; early stopping (patience 6) on the validation macro-F1.
  (W3: read experiments/train.py DEFAULTS and experiments/mil_train.py DEFAULTS for learning rate / weight decay before quoting them.)

### Method campaign on internal model-disjoint folds (pool = train + validation of split_platform: 168,588 windows, 2,810 h, 117 car models; 57 candidate models; 3 folds x 19 held-out models; fleet rule pins the one identity group spanning >= 5 car models (33 vehicles, 19 car models, about 480 h, 13.3 % of the pool windows) to the training side)
- best epochs of the reference recipe on the folds: 1-8 (median about 4) against 16-22 when early stopping on the shared-model
  validation split: that validation rewards memorising car models. Final models therefore train for a fixed number of epochs =
  median of the fold best epochs (GRU S9 11, ResNet S9 16, GRU B5 2, ResNet B5 7, MIL 10).
- G1 (reference recipe, fold drive_full, mean of 3 folds): size GRU 65.1 (65.3 / 69.5 / 60.4), ResNet 66.6; BEV vs ICE GRU 91.9
  (89.4 / 90.8 / 95.4), ResNet 93.1 - all within 5 points of the published test values.
- size sweep, fold 0, GRU, one seed (drive_full): S0 65.3; S1 car-model-balanced sampling 62.8; S2 + footprint head 66.8; S3 + CORAL
  64.8; S4 + mass and powertrain-4 heads 65.2; S5 + augmentation 66.2; S6 + same-class cutmix 64.8; S7 + DAS down-weighting 65.6;
  S8 + conditional adversarial car-model head 66.5; S9 + group-DRO-lite 66.9 (spread 4 points, like the fold-to-fold spread).
  Three folds: S0 65.1, S2 65.5, S5 66.5, S9 67.4 (66.9 / 70.5 / 64.8); ResNet S9 66.2 (67.0 / 70.4 / 61.2). G2 size (>= 70) fails.
- BEV vs ICE sweep, fold 0, GRU: B0 89.4; B1 88.0; B2 87.4; B3 88.7; B4 (+ augmentation) 92.8; B5 (+ DAS down-weighting) 92.8. Three
  folds: B0 91.9, B4 92.8, B5 93.1 (92.8 / 91.1 / 95.4); ResNet B0 93.1, ResNet B5 95.2 (95.7 / 93.9 / 96.1). G2 BEV vs ICE passes.
- leave-one-car-model-out probes with S9 (drive F1 of the single held-out model): Honda Odyssey 74.7, Hyundai Palisade 27.6,
  Hyundai Sonata 18.3, Tesla Model 3 11.3 (models near a threshold are systematically misread).
- episode MIL on the folds (gated attention, own bag prediction, drive_full): size 67.8 / 65.6 / 67.5 (mean 67.0 = S9's 67.4: G3 size
  fails); BEV vs ICE 96.0 / 98.2 / 93.2 (mean 95.8, +2.7 over B5: G3 BEV vs ICE passes). Window-mean aggregation of the same model:
  size 61.1, BEV vs ICE 88.8. Fold-0 ablation, gated attention vs learned mean of instance logits (own bag prediction): size 67.8
  vs 66.7; BEV vs ICE 96.0 vs 94.8.
- aggregation (out of fold, reference recipe, GRU + ResNet pooled): size drive_full mean 67.5 -> best variant 67.8 (+0.3),
  vehicle 60.5 -> 61.6; BEV vs ICE drive 93.0 -> DAS-down-weighted mean 94.8 (+1.8), vehicle 93.4 -> 95.5. Rule needs >= +1.0 on
  both tasks: the plain mean stays the headline. Ensembles out of fold: GRU + ResNet mean gains under one point over the best
  member on both tasks (BEV vs ICE 93.9 against 93.1-93.5; size 68.8 against 68.2) and stacking is worse (92.3; 66.5): no
  ensemble in the final list (G4 fails).

### Final models (split_platform test, 3 seeds; window test_full / drive_full / vehicle_full; [CI] = 95 % cluster bootstrap over car models, mean of the seeds' bounds; std over seeds where given)
- size: GRU S9 53.2+-0.3 / 67.1+-0.0 [54.4, 74.2] / 60.0+-0.6 [47.6, 66.7]; ResNet S9 53.2+-0.8 / 66.8+-1.7 [53.3, 74.6] / 62.6+-1.4 [49.0, 69.8];
  episode MIL 54.2+-0.3 / 64.6+-0.5 [50.7, 73.1] (own bag prediction 64.3) / 62.3+-0.9 [49.5, 71.0] (own bag 62.4).
  drive_human_only: GRU S9 61.7, ResNet S9 62.9, MIL 62.5 (bag 63.2). vehicle_full_multi: GRU S9 63.5, ResNet S9 66.1, MIL 60.0.
  Window test_balanced: GRU S9 50.1, ResNet S9 50.3, MIL 51.0; window human-only: 47.8, 49.5, 50.0. Validation (shared models):
  GRU S9 60.7, ResNet S9 62.8, MIL 79.1. Gain over the reference GRU at drive level: +4.2 (62.9 -> 67.1); window level unchanged.
- BEV vs ICE: GRU B5 77.0+-0.3 / 92.6+-0.1 [87.1, 96.2] / 93.4+-0.4 [87.9, 97.2]; ResNet B5 79.9+-0.7 / 93.0+-0.8 [87.2, 96.6] / 92.8+-0.4
  [86.5, 97.3]; episode MIL 78.8+-0.1 / 90.2+-0.2 [83.2, 94.7] (own bag prediction 93.0) / 90.4+-0.5 [82.4, 95.1] (own bag 92.8).
  drive_human_only: GRU B5 89.1, ResNet B5 88.7, MIL 89.9 (bag 91.5). vehicle_full_multi: GRU B5 95.5 [91.2, 98.7], ResNet B5 94.4,
  MIL 91.4. Window test_balanced: GRU B5 77.4, ResNet B5 80.2, MIL 78.5; window human-only: 83.7, 83.2, 82.2. Validation: GRU B5
  75.6, ResNet B5 79.3, MIL 97.5. Best ResNet B5 run per-class recall on windows: BEV 77.0, ICE 85.6. The MIL fold advantage
  (95.8) did not carry to the test models (93.0 own bag).
- supplementary aggregation on the final S9 / B5 models (pre-specified: log-odds mean with weights (1 - DAS share + 0.1)^0.5 x
  (1 + acceleration + deceleration episodes)): size drive_full 66.9 -> 68.7 (+1.7, 6 of 6 runs improved), BEV vs ICE 92.8 -> 93.5
  (+0.7, 6 of 6); headline stays the plain mean as pre-registered.
- held-out devices (split_device; window / drive_full [CI] / vehicle_full): size GRU S9 62.9+-1.0 / 79.5+-0.7 [70.8, 84.4] / 75.8+-0.7;
  ResNet S9 64.8+-0.3 / 80.6+-0.1 [71.4, 85.5] / 76.4+-0.8; MIL (1 seed) 68.4 / 78.8 / 79.2; drive_human_only 75.6 / 76.2 / 73.5.
  BEV vs ICE GRU B5 78.8+-0.4 / 95.1+-0.3 [90.9, 97.5] / 96.3+-0.0; ResNet B5 83.1+-0.2 / 97.2+-0.3 [94.7, 98.7] / 99.0+-0.4 [97.0, 100.0];
  MIL (1 seed) 86.5 / 96.1 (own bag 98.6) / 97.0; drive_human_only 96.2 / 97.4 / 97.9. Note: under split_device the S9 / B5 window
  scores are below the reference recipe (e.g. size GRU 62.9 vs 70.2) while the drive scores are above it (79.5 vs 78.1): the
  recipes trade within-model recognition for cross-model generalisation.

### Ceilings (zero training; size task, split_platform test; post hoc on the saved predictions of the published reference models)
- per-car-model majority oracle (every drive of a held-out model gets the model's majority prediction): reference ensemble of the
  12 deep runs 63.2 -> 72.0; GRU 62.9 -> 72.4; Transformer 61.8 -> 76.3; ResNet 61.1 -> 64.1; Mamba-2 61.2 -> 64.0; LLM + statistics
  GBM 64.9 -> 74.0. BEV vs ICE: reference 89.8 -> 93.5; ResNet 91.1 -> 94.0; GRU 89.0 -> 92.6.
- footprint-noise oracle (true footprint + per-drive noise sigma, class cut at the thresholds; 200 repeats): sigma 0.05 -> 97.1,
  0.10 -> 92.1, 0.15 -> 86.6, 0.20 -> 80.7, 0.30 -> 70.7; a per-car-model bias tau gives the same curve (0.05 -> 97.4, 0.10 -> 92.2,
  0.15 -> 87.3, 0.20 -> 80.5, 0.30 -> 69.8). Reaching 90 needs an error of about +-0.12 m^2 (about 2.5 % of a 4.6 m^2 footprint).
- curb-mass oracle: two mass cuts fitted on 6,364 training drives (117 car models; training macro-F1 81.7) score 63.3 (accuracy 69.4)
  on the 1,139 test drives with a known mass.
- separability: 21 of the 41 held-out car models lie within 0.15 m^2 of a differently-labelled training model. LARGE held-out
  unibody vehicles are predicted MID or SMALL: Genesis G80 0 % of drives correct, Genesis G90 0 %, Volkswagen Atlas 0 %, Tesla
  Model S 5.6 %, Kia EV9 3.3 %, Genesis GV80 29.7 %, Toyota Sienna 36.4 %; pickups and body-on-frame SUVs are recognised: GMC Yukon
  100 %, Ford Ranger 100 %, Ford F-150 Lightning 95.5 %. MID models just above 4.43 m^2 are read as SMALL: GWM Haval H6 0 %, Audi Q5
  0 %, Lexus RX 0 %, Nissan Altima 0 %, Kia Stinger 25 %. Window-level accuracy by distance of the footprint to the nearest threshold:
  53.0 % within 0.1 m^2, 59.6 % at 0.1-0.2, 75.4 % at 0.2-0.4.
- evidence sufficiency (reference predictions): drive accuracy by number of windows per drive stays between 63.8 and 73.8 % for
  size (bins 1-2 to > 64 windows) and between 85.7 and 93.2 % for BEV vs ICE; subsampling m windows per drive on drives with >= 32
  windows: size macro-F1 52.6 (m = 1) -> 54.9 (m = 32), all windows 56.0; BEV vs ICE 79.7 -> 89.3, all windows 89.1, saturating by
  8-16 windows. Accuracy also does not grow with the number of acceleration / deceleration episodes or lead-acceleration events.

### LLM-assisted feature discovery
- six proposer agents, one physics lens each (longitudinal powertrain, transmission signature, braking / regeneration / coasting,
  lateral inertia and geometry, car-following interaction, speed-dependent capability), saw only the channel definitions and a
  training-only sample of 2,727 windows; 62 features proposed, plus 12 from a gap-filling round = 74. Six verifier agents ran the
  code, fixed bugs, measured class separation (AUROC BEV vs ICE, Spearman with footprint) and dropped 21 features (coverage below
  the floor, degenerate or duplicate); an auditor excluded 19 (6 in the first audit, 13 in the second: duplicates, a v^2 speed
  confound, cues that come from the automation controller (lead-response gain and lag, braking-onset time-to-collision), recording
  artefacts); 34 registered. The first pass used a 40 % coverage floor (12 features reached the auditor); the floor was then lowered
  to 10 % because gradient boosting treats a missing regime as information, re-admitting 29. Everything fixed before any
  evaluation split was touched; features feed the same gradient-boosting / logistic-regression baselines.
- most important (permutation importance on the validation split): BEV vs ICE: lift-off deceleration while moving, lift-off
  deceleration shortly after pedal release, share of deceleration under a pressed accelerator, share of deceleration without the
  brake, lead-deceleration-to-ego-deceleration delay, 10th percentile of brake-free deceleration; size: curvature-per-steering-angle
  gain at 6-18 m/s and at low speed, Ackermann gain at 6-12 m/s, lead-deceleration-to-ego-deceleration delay.
- results: see first round (LLM features alone: size 40.5 window / 63.2 drive; BEV vs ICE 69.6 / 89.4, human-only window 84.2; with
  statistics: size 54.1 / 64.9, BEV vs ICE 73.2 / 80.9). Under split_device LLM + statistics + GBM is the best size model (72.4 / 80.6).

### Episodes and the MIL model
- 735,571 episodes from 8,662 drives (545 drives have none): ACC 230,147; DEC_LIFT 129,253; DEC_BRK 95,844; STOP 51,568; LAUNCH 51,078;
  LEAD_ACC 22,370; LEAD_DEC 7,166; TURN 148,145. Share of episodes without automation activity (strict, 10 s guard): ACC 32.6 %,
  DEC_LIFT 19.4 %, DEC_BRK 48.4 %, STOP 42.0 %, LAUNCH 41.9 %, LEAD_ACC 27.3 %, LEAD_DEC 12.5 %, TURN 47.8 %.
- detection: speed smoothed by a Savitzky-Golay filter (5 samples, order 2); acceleration / deceleration episodes = monotone
  segments (tolerance 0.3 m/s) of >= 3 s with |dv| >= 2 m/s and mean |a| >= 0.1 m/s^2, 2 s of cruise trimmed at both ends;
  DEC_LIFT = deceleration with the brake pressed in at most 10 % of its samples, DEC_BRK otherwise; STOP = standstill of >= 1 s
  reached from >= 2 m/s within 15 s; LAUNCH = departure from standstill; LEAD_ACC = lead acceleration slope >= 0.5 m/s^2 held 1 s after
  2 s below 0.3 m/s^2 in car-following range (headway <= 6 s), ego response = a >= 0.3 m/s^2 for 0.5 s within 8 s; LEAD_DEC = lead slope
  <= -0.6 m/s^2; TURN = |curvature| >= 0.01 1/m for >= 1 s at 2-12 m/s.
- representation: 18-s crops (180 samples) anchored at the onset (onset at sample 40; STOP: standstill onset at sample 140), 22
  channels (20 official + smoothed acceleration + validity); ACC / DEC episodes also as a speed-indexed grid (0-36 m/s in 0.5 m/s
  steps, 72 bins x 6 channels: smoothed acceleration, pedal flags, automation flag, validity, time from onset); 9-dimensional
  context (anchor speed, dv, duration, mean and peak acceleration, automation share, lead share, headway, gap flag).
- model: shared episode encoder (convolutional stem + residual blocks, type embedding, context MLP, d = 128), speed-grid encoder,
  adapter for frozen 60-s window embeddings of the published window models (instances of a ninth type), gated attention pooling
  per task (size, BEV vs ICE), heads for size, BEV vs ICE, four-class powertrain, log footprint, log curb mass and two ordinal
  thresholds; about 0.99 M parameters. Loss = bag cross-entropies (label smoothing 0.05, masked by task eligibility) + 0.3 x
  instance cross-entropies + 0.5 Huber(log footprint) + 0.25 Huber(log mass) + 0.25 CE(powertrain-4) + 0.25 BCE(ordinal).
  Bags: up to 64 episodes (lead and turn episodes first, each capped at 16) + 8 window embeddings per drive in training, 128 + 16 at
  evaluation; batches of 24 drives with a cap on the share of one car model; augmentation: lateral sign flip (p 0.5), instance
  dropout 0.2, lead dropout 0.1, noise 0.02. AdamW, warm-up + cosine; fixed 10 epochs in the final run. The drive-level output
  is the bag prediction; window-mean aggregation of the same model is reported as the comparable drive_full row.

### Recipes S9 / B5 (window models)
- S1 / B1: sampling by car model inside each class (weights hours^0.5 per model, capped); S2 / B2: footprint regression head (SmoothL1
  on the standardised log footprint); S3: CORAL ordinal head (two cumulative thresholds); S4 / B3: log curb-mass head and four-class
  powertrain head (B3 also trains on HEV / PHEV windows without a BEV-vs-ICE label); S5 / B4: physics-safe augmentation = sign flip
  of the six lateral channels together (p 0.5), one masked span of 3-15 s with lead_valid set to 0 (p 0.5), dropout of one channel
  group (lead / steering torque / pedals; p 0.1), Gaussian noise 0.02 on continuous channels; no speed or amplitude scaling and no
  time reversal (both change the vehicle signature); S6: same-class cutmix across car models; S7 / B5: DAS down-weighting with
  sample weight (1 - beta) + beta (1 - DAS share), beta 0.5; S8: conditional adversarial car-model head (gradient reversal, lambda
  0.1); S9: group-DRO-lite (per-car-model loss re-weighting). Head loss weights: classification 1, CORAL 0.5, footprint 1, mass 0.5,
  powertrain-4 0.3. Decision rule: softmax arg-max (footprint-bucket and CORAL decisions did not win on the folds).


### Additional verified numbers used in the sections (added after the first writing pass)
- hold ages: lead (radar, 4 Hz) 0.75 s, pose 0.625 s, controller topics 0.3 s; gap flag when the inter-sample interval exceeds 0.3 s;
  moving = speed >= 1 m/s; windows need >= 20 % moving samples; low-support cell = under 10 test hours, 5 groups or 2 platforms;
  balanced list caps: one group <= 25 %, one platform <= 50 % of a class; class-weight cap 10.
- training: warm-up over the first 3 % of the steps; 180 window statistics for the feature baselines (9 per channel).
- feature baselines on split_platform (window human-only / drive human-only / vehicle_full): statistics + GBM size 46.3 / 50.5 / 51.5,
  BEV vs ICE 73.5 / 74.3 / 76.0; LLM + GBM size 47.3 / 58.9 / 58.6, BEV vs ICE 84.2 / 89.1 / 90.1; LLM + statistics + GBM size
  49.8 / 64.2 / 61.8, BEV vs ICE 80.2 / 82.6 / 83.4.
- vehicle_full of the reference sequence models (split_platform): size ResNet 56.4, GRU 57.2, Transformer 55.3, Mamba-2 55.2;
  BEV vs ICE ResNet 90.3, GRU 88.6, Transformer 89.0, Mamba-2 90.6.
- BEV vs ICE fold pool: 121,381 windows, 2,023 h, 107 car models, 51 candidates; folds hold out 16 / 15 / 20 models; fleet group
  17.1 % of that pool.
- gates: G2 BEV vs ICE = fold drive_full >= 90 and >= +1 over the reference recipe (or drive_human_only >= 92); G3 = >= +2 over
  the best window recipe (size) or >= 91 (BEV vs ICE); G4 = >= +1.0 over the best single model.
- evidence-sufficiency subsampling uses 50 repeated draws per m; window accuracy by distance to the nearest threshold (reference
  GRU run) also 55.2 % at 0.4-0.8 m^2 (bin of the Kia EV9 and Kia Carnival, footprint 5.30) and 89.9 % beyond 0.8 m^2.
- region share of the released hours: North America 83.8 %, Europe 4.8 %, Asia 4.2 %, other / unknown 7.2 % (continent-level buckets).
- Fig. 2 annotations regenerated with the reference-ensemble values: achieved 63.2, per-car-model majority oracle 72.0.

## Figures and tables (files exist; use these labels)
- Fig. 1 `figures/evidence_sufficiency.pdf` label fig:evidence — drive-level macro-F1 vs windows sampled per drive.
- Fig. 2 `figures/footprint_noise_oracle.pdf` label fig:ceiling — footprint-noise oracle with achieved and majority-oracle lines.
- Table I dataset summary (W2 writes it inline). Table II main results split_platform (W4; rows: statistics + GBM;
  LLM + statistics + GBM; ResNet1D; bi-GRU; Transformer; Mamba-2; GRU S9 / ResNet S9 (size); GRU B5 / ResNet B5 (BEV vs ICE);
  episode MIL; columns: window full, window human-only, drive full, drive human-only, vehicle full). Table III held-out
  devices (compact). Table IV folds / recipe sweep (compact). Table V LLM features (compact).
- LaTeX: `booktabs` is loaded in main.tex; use `\begin{table*}` only for Table II; `\si` is not available (write units plainly).

## Citation keys available in refs.bib (use only these; anything else becomes a `% TODO cite` comment)
gu2023mamba, dao2024mamba2, vaswani2017attention, cho2014gru, wang2017tsc, ilse2018mil, cao2020coral, sagawa2020dro,
ganin2016dann, geirhos2020shortcut, pedregosa2011sklearn, schafer2018comma2k19, wang2026adasto, tan2024llmts,
savitzky1964, loshchilov2019adamw, romeraparedes2024funsearch, ismailfawaz2019review, ke2017lightgbm, breiman2001rf, long2018cdan

## Style rules (binding)
- Write "driving automation system (DAS)" at first use and "DAS" afterwards; never the four-letter driver-assistance
  acronym (except inside the title of the cited dataset paper, which is quoted as is in the bibliography only).
- No identifiers, no device or route ids, no locations, no dates of recordings; car model names are allowed.
- Claims hedged and tied to the protocol ("on held-out car models", "drive-level"); results are reference points, not a
  leaderboard; do not call the methods novel; describe the Mamba-2 implementation as a pure-PyTorch implementation of the
  published algorithm.
- Numbers only from this file (copy them; do not round further except where the outline already shows one decimal).
- Each section file starts with `\section{...}` (or the abstract environment) and contains only its own section; use
  `\label{sec:...}` names: intro, related, dataset, methods, experiments, discussion, conclusion.
- Write in plain, direct English; short paragraphs; no bullet lists in the prose except where a list of channels or
  contributions is natural (use the IEEE `itemize` sparingly).

### External validation (verified against experiments/transfer/RESULTS.md on 11 Nov 2026)
Zero-shot: the final `split_platform` checkpoints (size bi-GRU S9, BEV vs ICE 1-D ResNet B5, seed ensembles) and the channel-subset
variants of experiments/transfer/VARIANTS.md (V1 lateral + speed = 10 channels, V3 speed-only = speed, acceleration, standstill;
same recipes, 2 seeds) run unchanged on public sets; vehicle level = device / archive / car; 95 % intervals = cluster bootstrap over
vehicles. Used in sections/05_experiments.tex (subsection "External Validation on Public Datasets", Table VI `tab:external`),
00_abstract.tex and 07_conclusion.tex. Only favourable findings are in the paper (owner's instruction, 11 Nov 2026); the
unfavourable ones stay in RESULTS.md, which the paper points to with one neutral sentence. Source section of RESULTS.md in brackets.
- commaCarSegments, device-disjoint selection (`ccs`) [2.12, 1.1]: 45,763 one-minute windows, 762.7 h, 4,061 devices absent from the
  lab library, 113 car models, 161 openpilot platforms; 20 channels decoded from CAN with opendbc where the platform provides them
  (torque absent on 15.4 %, lead on 55.7 % of the windows); yaw / curvature / lateral acceleration derived from the steering angle
  with a bicycle model. Understeer correction [2.15, 3.7, UNDERSTEER.md]: one global k = 0.00079289 s^2/m^2 (paper: 0.00079), the
  hours-weighted median over 62 train / validation platforms fitted on the lab's own IMU / pose yaw, never on external labels.
- ccs powertrain, full 20-channel model, as-run decode [2.12 (a), 0]: 86 of 88 labelled car models, 97.7 % [96.2, 100.0]; 87.8 %
  [86.6, 89.1] of 2,512 labelled devices; held-out stratum (29 of the lab's 41 test models): 27 of 27 labelled models, 94.2 %
  [92.0, 96.2] of 463 devices. Corrected yaw [2.15 (a)]: 88.9 % of devices (paper states both). Speed-only V3 [2.12 (b), 0]:
  92.6 % [91.5, 93.6] of devices, 87 of 88 models. Not used in the paper (dropped for length, available): device accuracy by minutes
  per device 80.1 % (one window) -> 96.3 % (10+) [2.12 table f]; hybrids in between, windows called BEV ICE 17.5 / HEV 32.0 /
  PHEV 44.6 / BEV 78.7 % [2.12, 3.6 point 5].
- ccs size, full model, corrected yaw [2.15 (a), 3.7]: 63.2 % of 4,061 devices, macro-F1 60.3 (internal held-out vehicle level
  63.5 / 59.7); 72 of 113 car models; LARGE device recall 51.9 % (532 of 1,025; internal 50.6); SMALL 1,594 of 1,762 devices (not in
  the paper); recovered LARGE platforms: all four Pacifica, Palisade, CX-9, Sienna 4th gen, both Escalades, Rivian R1, Maverick,
  Ranger, G80, Hyundai Genesis, Crafter, plus the pickups already right (paper: "Pacifica, Palisade, CX-9, Escalade and
  fourth-generation Sienna platforms and the pickups").
- ccs, nine recovered Honda platforms (`ccs_honda`, as-run decode) [2.13, 0]: 2,132 windows, 35.5 h, 230 devices, 5 car models;
  SMALL 226 of 230 devices (98.3 % [96.5, 99.6]), 5 of 5 models; ICE 227 of 229 devices (99.1 % [97.8, 100.0]), 4 of 4 models;
  held-out HR-V 5 of 5 on both tasks.
  (Paper text and Table VI mark these Honda rows "as-run decode"; the corrected-yaw Honda size number, 215 of 230 [2.15 (a)], is not in the paper.)
- commaSteeringControl (`csc`) [2.10, 1.1]: 36 platform archives (2023 release, lateral controller active, highway), 51,462 windows,
  857.7 h, 25 car models, one archive = one pseudo-vehicle (no device ids), V1 channel list (speed, CAN aEgo, steering, localizer
  lateral channels; no lead / torque / pedals). Powertrain, V3 speed-only: 35 of 36 archives, 97.2 % [91.7, 100.0], macro-F1 96.1;
  model level 96.0 % of 25 = 24 of 25; held-out stratum 9 of 10 archives. Size, V1 lateral + speed: 63.9 % [50.0, 80.6] archive
  accuracy, macro-F1 62.3; internal V1 vehicle reference 62.6 / 58.0; 64.0 % of 25 models = 16 of 25.
- MoreTorque/rlogs (`moretorque`) [2.11, 0]: 12 cars, 7,190 windows, 119.8 h, all 20 channels, community openpilot forks; size:
  all 8 SMALL cars SMALL (vehicle recall SMALL 100.0; Mazda 3 7 of 7); powertrain: all 11 ICE cars ICE.
- VED (`ved`) [2.7, 0]: 384 cars, 1 Hz OBD speed, V3 speed-only: 258 of 264 ICE cars called ICE (97.7 %).
- OpenACC one-vehicle campaign (`openacc_500x`) [2.6, 0]: one Fiat 500X, 20 drivers, 429 trips (13,523 windows, 225.4 h), V3
  speed-only: 99.1 % of the trips ICE (425 of 429).
- Available but not in the paper (length): OpenACC platoon campaigns [2.2] V2 longitudinal-only calls all 7 BEV cars BEV (Model 3
  x2, Model S, Model X, e-tron, I-Pace, e-Golf); phase-1 pooled 23 of 24 named BEV cars outside VED [0]; Yuan et al. Tesla Model 3
  80.2 % of 106 urban trips BEV [2.5] (no verified bib entry requested).
- Bib entries added (refs.bib): commasteeringcontrol, commacarsegments, moretorque (Hugging Face URLs), oh2019ved (arXiv:1905.02081),
  makridis2021openacc (TR-C 125, 103047). Removed from the bibliography by the trims: breiman2001rf and tan2024llmts are no longer
  cited (entries kept in refs.bib).
