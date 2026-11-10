# Channel-subset variants of the final models (our data, held-out car models)

Written 9 Nov 2026 by experiments/transfer/train_variants.py --render.

Macro-F1 in percent on the `split_platform` test split (car models held out), mean ± std over seeds; in brackets the difference to the full model. `window` = one prediction per 60-s window (`test_full`), `drive` = mean class probability over the windows of a drive, `vehicle` = the same over all test drives of a vehicle (docs/PROTOCOLS.md). The full models are the final runs of experiments/final_manifest.json (3 seeds, the numbers of experiments/RESULTS.md); every variant re-trains the same recipe with the same fixed epoch count and kwargs, seeds 0, 1, with every channel outside its list set to NaN (= the missing value 0) on train, validation and test (`experiments.train.run(channels=...)`). Chance level is about 33 (size) and 50 (BEV vs ICE).

| variant | inputs kept | masked |
|---|---|---|
| V1 lateral + speed (10 of 20) | `v_mps, a_mps2, standstill, das_enabled, steer_angle_deg, steer_rate_degps, yaw_rate_radps, curv_1pm, lat_acc_mps2, steer_pressed` | lead group, steering torque, gas / brake pedal flags; external sets: commaSteeringControl |
| V2 longitudinal-only (11 of 20) | `v_mps, a_mps2, standstill, das_enabled, lead_valid, lead_d_m, lead_v_rel_mps, lead_speed_mps, lead_a_slope_mps2, thw_s, ttc_s` | every lateral channel (steering angle / rate / torque, yaw rate, curvature, lateral acceleration), steer_pressed, pedal flags; external sets: ULTra-AV, OpenACC platoons, UGA EV platoon, microSIM-ACC |
| V3 speed-only (3 of 20) | `v_mps, a_mps2, standstill` | everything but speed, acceleration and the standstill flag (no automation flag, no lead, no lateral channels, no pedals); external sets: Yuan et al. Tesla Model 3, VED, OpenACC one-vehicle campaign |

## size (SMALL / MID / LARGE), bi-GRU recipe S9, 11 epochs

| model | inputs | window full | window balanced | window human-only | drive full | drive human-only | vehicle full | val | seeds |
|---|---|---|---|---|---|---|---|---|---|
| full model (gru S9) | 20 | 53.2 ± 0.3 | 50.1 ± 0.3 | 47.8 ± 0.3 | 67.1 ± 0.0 | 61.7 ± 0.3 | 60.0 ± 0.6 | 60.7 ± 1.2 | 3 |
| V1 lateral + speed | 10 | 54.5 ± 0.7 (+1.3) | 51.6 ± 0.4 (+1.5) | 53.7 ± 0.5 (+5.9) | 66.1 ± 0.9 (−1.0) | 65.2 ± 0.3 (+3.5) | 59.5 ± 1.6 (−0.5) | 61.3 ± 1.0 (+0.5) | 2 |
| V2 longitudinal-only | 11 | 34.8 ± 0.0 (−18.4) | 33.3 ± 0.0 (−16.8) | 33.4 ± 0.1 (−14.5) | 33.4 ± 0.4 (−33.7) | 36.0 ± 0.1 (−25.6) | 33.2 ± 0.0 (−26.9) | 38.5 ± 0.0 (−22.3) | 2 |
| V3 speed-only | 3 | 40.7 ± 0.7 (−12.5) | 40.2 ± 0.6 (−9.9) | 41.3 ± 0.4 (−6.6) | 45.3 ± 0.9 (−21.7) | 46.7 ± 0.8 (−15.0) | 45.7 ± 0.9 (−14.4) | 48.9 ± 0.6 (−11.9) | 2 |

Per-class recall of the drive-level decision (`drive full`, mean over seeds):

| model | SMALL | MID | LARGE |
|---|---|---|---|
| full model | 80.1 | 61.2 | 58.7 |
| V1 lateral + speed | 83.7 | 49.5 | 62.5 |
| V2 longitudinal-only | 70.6 | 10.8 | 22.1 |
| V3 speed-only | 72.7 | 37.2 | 25.5 |

## BEV vs ICE, 1-D ResNet recipe B5, 7 epochs

| model | inputs | window full | window balanced | window human-only | drive full | drive human-only | vehicle full | val | seeds |
|---|---|---|---|---|---|---|---|---|---|
| full model (resnet B5) | 20 | 79.9 ± 0.7 | 80.2 ± 0.8 | 83.2 ± 0.6 | 93.0 ± 0.8 | 88.7 ± 1.0 | 92.8 ± 0.4 | 79.3 ± 0.2 | 3 |
| V1 lateral + speed | 10 | 67.3 ± 0.0 (−12.6) | 68.6 ± 0.3 (−11.6) | 66.6 ± 0.7 (−16.6) | 74.2 ± 0.5 (−18.7) | 69.8 ± 0.5 (−18.9) | 74.2 ± 1.0 (−18.6) | 73.4 ± 0.0 (−6.0) | 2 |
| V2 longitudinal-only | 11 | 73.0 ± 0.4 (−6.9) | 72.3 ± 0.2 (−7.9) | 71.2 ± 0.7 (−12.0) | 88.8 ± 0.3 (−4.1) | 83.8 ± 0.7 (−4.9) | 89.8 ± 0.7 (−3.0) | 71.3 ± 0.2 (−8.0) | 2 |
| V3 speed-only | 3 | 73.8 ± 0.1 (−6.1) | 72.5 ± 0.2 (−7.7) | 72.2 ± 0.5 (−11.0) | 88.1 ± 0.2 (−4.8) | 85.2 ± 0.3 (−3.4) | 91.0 ± 0.0 (−1.8) | 74.4 ± 0.2 (−4.9) | 2 |

Per-class recall of the drive-level decision (`drive full`, mean over seeds):

| model | BEV | ICE |
|---|---|---|
| full model | 87.9 | 96.9 |
| V1 lateral + speed | 71.6 | 77.5 |
| V2 longitudinal-only | 78.3 | 97.1 |
| V3 speed-only | 78.4 | 95.9 |

## Diagnostics (size task, seed 0): why longitudinal-only (V2) scores below speed-only (V3)

Same recipe (S9) unless stated; one seed, so differences below about 2 points are noise. The last column is the training cross-entropy as the recipe weights it (group-DRO re-weights the samples, so the no-DRO run's value is not on the same scale).

| run | inputs | window full | drive full | vehicle full | val | train cls loss (last epoch) |
|---|---|---|---|---|---|---|
| V2 longitudinal-only | 11 | 34.8 ± 0.0 | 33.4 ± 0.4 | 33.2 ± 0.0 | 38.5 ± 0.0 | 0.883 |
| V3 speed-only | 3 | 40.7 ± 0.7 | 45.3 ± 0.9 | 45.7 ± 0.9 | 48.9 ± 0.6 | 0.839 |
| V2nodro V2 channels, no group-DRO (dro=False) | 11 | 38.1 | 42.7 | 40.5 | 49.5 | 0.692 |
| V3das V3 + das_enabled | 4 | 37.6 | 39.8 | 36.3 | 44.6 | 0.904 |
| V3lead V3 + lead group | 10 | 36.3 | 35.2 | 31.1 | 40.4 | 0.897 |

## What each variant loses

* **V1 lateral + speed**: Nothing is lost for the size task (54.5 / 66.1 / 59.5 window / drive / vehicle against 53.2 / 67.1 / 60.0; the human-only window score even rises by 6 points), so the size signal lives in the lateral channels (steering geometry at a given speed and yaw) and the lead, torque and pedal groups are dispensable for it. For BEV vs ICE the variant loses the pedal flags, i.e. the direct view of deceleration without a brake application (regenerative braking), and drops to 67.3 / 74.2 / 74.2 (-12.6 / -18.7 / -18.6); both classes suffer (drive-level recall BEV 71.6, ICE 77.5), so commaSteeringControl powertrain numbers must be read against roughly 74, not 93.
* **V2 longitudinal-only**: For the size task the variant is at chance (34.8 / 33.4 / 33.2; drive-level recall MID 10.8, LARGE 22.1): without the lateral channels the recipe cannot learn size, and the lead group and the automation flag make it worse than the speed-only inputs (diagnostics below), so size results on longitudinal-only external sets are not interpretable with this variant; use V3 as the reference there. For BEV vs ICE most of the accuracy survives (73.0 / 88.8 / 89.8, -6.9 / -4.1 / -3.0): what is lost is BEV recall at the drive level (78.3 against 87.9, ICE unchanged at 97.1), the share of BEV drives whose regenerative deceleration is not visible without the pedal flags.
* **V3 speed-only**: Speed and acceleration alone keep a weak but real size signal (40.7 / 45.3 / 45.7, chance 33; SMALL 72.7, MID 37.2, LARGE 25.5 drive-level recall), and for BEV vs ICE they keep nearly everything the longitudinal set keeps (73.8 / 88.1 / 91.0, -6.1 / -4.8 / -1.8; BEV recall 78.4, ICE 95.9): the lead group adds nothing to the powertrain decision, the loss against the full model is the missing pedal flags. Speed-only external sets (Yuan Tesla, VED, one-vehicle OpenACC) are therefore meaningful for the powertrain task and only indicative for size.

## Reading external results with these numbers

* The variants reuse the final recipes unchanged (same kwargs, same fixed epoch counts chosen for the 20-channel models, two seeds, no re-tuning), so they measure what the recipe loses with fewer inputs, not the best reduced-input model; the longitudinal size variants in particular were still improving at the last epoch.
* External size results: compare V1 datasets with the full-model numbers (V1 is as good as the full model); for longitudinal or speed-only datasets use V3 and its 45-46 drive / vehicle reference, and do not use V2 for size.
* External powertrain results: V2 and V3 datasets are compared with about 88-91 at drive and vehicle level (73-74 per window); V1 datasets with 74 (67 per window). Window-level numbers fall more than drive-level numbers in every variant, so external sets that only provide isolated 60-s windows (commaCarSegments, commaSteeringControl) are penalised twice.
* The diagnostics show that under the model-disjoint split a channel can hurt: `das_enabled` and the lead group identify the driving-automation system and the lead sensor (radar or camera) of the training car models and let the network fit those models instead of the vehicle class, and group-DRO, which up-weights the worst-fitted car models, amplifies this. The same mechanism explains why the lateral channels hurt the powertrain variant V1 (steering geometry identifies the car model).

## Files

* checkpoints: `<data-root>/experiments/ckpt/transfer__<task>__<arch>__<Vk>__s<seed>.pt`; `cfg["channels"]` holds the variant's channel list (tensor order unchanged in `channels`), `experiments.regen_probs.load_checkpoint` rebuilds the network with its heads and a `Preprocess` that applies the same mask;
* results: `<data-root>/experiments/results/<tag>.json` (`channels` and `cfg.channels` = the list) and `<tag>_probs.npz` (window_id, probs, y of `test_full`); `experiments.run_all.load_results` leaves these channel-restricted runs out of RESULTS.md and the paper tables;
* script: `experiments/transfer/train_variants.py` (training and this page), training code: `experiments/train.py` (`channels`, `tag_prefix`), mask: `experiments/data.py` (`channel_keep`, `Preprocess(keep=)`, `Augment(keep=)`).
