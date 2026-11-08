# Oracle studies on the saved test predictions (post-hoc ceilings)

Every number on this page is computed after the fact from the saved test-split predictions of the already-published models (64 runs; `experiments/RESULTS.md`) and from the public index tables; nothing was trained for it. These are **ceilings and diagnostics, not results**: because they look at the test split, none of them may select a model, a rule or a hyper-parameter. Any idea they suggest has to win on the internal model-disjoint folds first (`experiments/CAMPAIGN.md`). Macro-F1 in percent; `drive_full` etc. follow docs/PROTOCOLS.md (drive = mean class probability over the drive's test windows, arg-max). Car-model names are public labels of the dataset; no drive or vehicle identifier appears here.

Aggregation rules: `mean` = mean probability (official); `logodds` = mean per-class log-odds (probabilities clipped at 1e-4); `median` = per-class median; `confidence` = mean weighted by (max p - 1/k); `trimmed` = 20 % trimmed mean per class; `vote` = majority of the per-window arg-max.

## size / split_platform

17 saved runs (9 model families), 26144 test windows, 1141 test drives with at least one window. Reference predictions for (d) and (e): mean probability of 12 deep runs.

### (a) Aggregation rule x window mask (drive macro-F1, mean over seeds)

| model | mean | logodds | median | confidence | trimmed | vote | seeds |
|---|---|---|---|---|---|---|---|
| feat_logreg | 48.4 | 48.3 | 46.8 | 48.2 | 47.5 | 46.8 | 1 |
| feat_hgb | 54.7 | 55.0 | 53.7 | 53.6 | 53.6 | 51.8 | 1 |
| feat_llm_logreg | 45.4 | 45.3 | 39.6 | 45.3 | 43.4 | 36.5 | 1 |
| feat_llm_hgb | 63.2 | 63.6 | 55.7 | 62.5 | 60.6 | 54.3 | 1 |
| feat_llmstats_hgb | 64.9 | 65.4 | 62.7 | 65.3 | 64.7 | 61.9 | 1 |
| resnet | 61.1 | 61.2 | 60.6 | 61.3 | 60.9 | 59.6 | 3 |
| gru | 62.9 | 63.5 | 62.4 | 62.9 | 62.6 | 61.9 | 3 |
| transformer | 61.8 | 61.9 | 61.8 | 62.0 | 62.0 | 61.0 | 3 |
| mamba2 | 61.2 | 61.4 | 60.6 | 61.2 | 60.9 | 59.8 | 3 |

Mean-probability rule under the window masks (drive macro-F1 (drives kept)): `all` = every test_full window; `moving` = moving_frac >= 0.5; `episodes` = n_accel_ep + n_decel_ep >= 1; `human` = human_only_strict.

| model | all | moving | episodes | human |
|---|---|---|---|---|
| feat_logreg | 48.4 (1141) | 48.7 (1133) | 48.4 (1136) | 43.6 (732) |
| feat_hgb | 54.7 (1141) | 54.0 (1133) | 55.6 (1136) | 50.5 (732) |
| feat_llm_logreg | 45.4 (1141) | 45.0 (1133) | 45.1 (1136) | 42.1 (732) |
| feat_llm_hgb | 63.2 (1141) | 62.8 (1133) | 62.6 (1136) | 58.9 (732) |
| feat_llmstats_hgb | 64.9 (1141) | 65.0 (1133) | 67.0 (1136) | 64.2 (732) |
| resnet | 61.1 (1141) | 61.0 (1133) | 62.0 (1136) | 60.3 (732) |
| gru | 62.9 (1141) | 62.9 (1133) | 64.3 (1136) | 63.3 (732) |
| transformer | 61.8 (1141) | 61.9 (1133) | 63.3 (1136) | 61.3 (732) |
| mamba2 | 61.2 (1141) | 61.3 (1133) | 62.4 (1136) | 61.1 (732) |

### (b) Ensembles (macro-F1)

| ensemble | members | window | drive (mean prob) | drive (vote over members) | drive (stacked) | members: mean drive | members: best drive |
|---|---|---|---|---|---|---|---|
| seeds: resnet | 3 | 55.7 | 60.3 | 60.6 | 64.5 | 61.1 | 62.5 |
| seeds: gru | 3 | 55.6 | 63.1 | 62.7 | 65.3 | 62.9 | 63.9 |
| seeds: transformer | 3 | 53.9 | 62.3 | 62.2 | 63.3 | 61.8 | 63.2 |
| seeds: mamba2 | 3 | 54.1 | 61.5 | 60.7 | 63.8 | 61.2 | 61.5 |
| architectures: deep, seed 0 | 4 | 55.1 | 62.8 | 61.2 | 65.1 | 61.4 | 62.1 |
| architectures: deep, all seeds | 12 | 55.6 | 63.2 | 62.5 | 65.8 | 61.8 | 63.9 |
| deep (all seeds) + feat_hgb | 13 | 56.1 | 60.2 | 63.0 | - | 61.2 | 63.9 |
| deep (all seeds) + feat_llmstats_hgb | 13 | 56.8 | 64.4 | 62.8 | - | 62.0 | 64.9 |
| all saved runs | 17 | 55.8 | 63.3 | 62.6 | - | 59.9 | 64.9 |

### (c) Vehicle pooling

| level | class / cell | drives | vehicles | single-drive vehicles | share single-drive (%) |
|---|---|---|---|---|---|
| class | LARGE | 251 | 77 | 32 | 41.6 |
| class | MID | 292 | 93 | 48 | 51.6 |
| class | SMALL | 598 | 159 | 73 | 45.9 |
| cell | LARGE | BEV | 70 | 20 | 10 | 50.0 |
| cell | LARGE | HEV | 17 | 9 | 4 | 44.4 |
| cell | LARGE | ICE | 164 | 48 | 18 | 37.5 |
| cell | MID | BEV | 127 | 29 | 15 | 51.7 |
| cell | MID | HEV | 49 | 16 | 8 | 50.0 |
| cell | MID | ICE | 110 | 43 | 21 | 48.8 |
| cell | MID | PHEV | 3 | 3 | 3 | 100.0 |
| cell | MID | unknown | 3 | 2 | 1 | 50.0 |
| cell | SMALL | BEV | 130 | 44 | 17 | 38.6 |
| cell | SMALL | HEV | 106 | 22 | 11 | 50.0 |
| cell | SMALL | ICE | 233 | 58 | 26 | 44.8 |
| cell | SMALL | PHEV | 55 | 12 | 4 | 33.3 |
| cell | SMALL | unknown | 74 | 26 | 17 | 65.4 |
| all | all | 1141 | 329 | 153 | 46.5 |

| model | drive_full | vehicle_full (vehicles) | vehicle_full_multi (vehicles) | seeds |
|---|---|---|---|---|
| feat_logreg | 48.4 | 42.4 (329) | 43.0 (176) | 1 |
| feat_hgb | 54.7 | 51.5 (329) | 53.4 (176) | 1 |
| feat_llm_logreg | 45.4 | 37.3 (329) | 39.0 (176) | 1 |
| feat_llm_hgb | 63.2 | 58.6 (329) | 61.5 (176) | 1 |
| feat_llmstats_hgb | 64.9 | 61.8 (329) | 62.1 (176) | 1 |
| resnet | 61.1 | 56.4 (329) | 56.9 (176) | 3 |
| gru | 62.9 | 57.2 (329) | 57.4 (176) | 3 |
| transformer | 61.8 | 55.3 (329) | 55.6 (176) | 3 |
| mamba2 | 61.2 | 55.2 (329) | 56.3 (176) | 3 |

### (d) Evidence sufficiency (reference predictions)

Drive accuracy with a Wilson 95 % interval by the amount of evidence in the drive (`n_windows` = selected windows; the other quantities are window-index counts summed over the drive's windows).

`n_windows`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 1-2 | 58 | 63.8 | 50.9-74.9 | 56.1 |
| 3-4 | 109 | 67.9 | 58.6-75.9 | 61.9 |
| 5-8 | 202 | 73.8 | 67.3-79.3 | 70.9 |
| 9-16 | 292 | 71.6 | 66.1-76.4 | 66.7 |
| 17-32 | 248 | 65.3 | 59.2-71.0 | 58.2 |
| 33-64 | 160 | 73.1 | 65.8-79.4 | 54.1 |
| >64 | 72 | 69.4 | 58.0-78.9 | 60.4 |

`n_accel_ep`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 10 | 70.0 | 39.7-89.2 | 45.2 |
| 1-2 | 41 | 68.3 | 53.0-80.4 | 58.0 |
| 3-4 | 56 | 69.6 | 56.7-80.1 | 60.9 |
| 5-8 | 158 | 74.7 | 67.4-80.8 | 71.3 |
| 9-16 | 291 | 70.8 | 65.3-75.7 | 66.2 |
| 17-32 | 336 | 71.1 | 66.1-75.7 | 65.0 |
| 33-64 | 186 | 61.8 | 54.7-68.5 | 50.6 |
| >64 | 63 | 73.0 | 61.0-82.4 | 58.0 |

`n_decel_ep`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 8 | 62.5 | 30.6-86.3 | 27.8 |
| 1-2 | 49 | 75.5 | 61.9-85.4 | 63.5 |
| 3-4 | 71 | 66.2 | 54.6-76.1 | 60.2 |
| 5-8 | 160 | 68.1 | 60.6-74.8 | 63.6 |
| 9-16 | 293 | 74.1 | 68.8-78.7 | 70.0 |
| 17-32 | 321 | 71.7 | 66.5-76.3 | 64.8 |
| 33-64 | 183 | 62.8 | 55.6-69.5 | 52.5 |
| >64 | 56 | 67.9 | 54.8-78.6 | 50.4 |

`n_lead_accel_events`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 843 | 70.2 | 67.1-73.2 | 63.3 |
| 1-2 | 243 | 68.7 | 62.6-74.2 | 62.7 |
| 3-4 | 35 | 71.4 | 54.9-83.7 | 68.4 |
| 5-8 | 16 | 75.0 | 50.5-89.8 | 48.0 |
| 9-16 | 3 | 33.3 | 6.1-79.2 | 16.7 |
| 17-32 | 1 | 100.0 | 20.7-100.0 | 100.0 |

Subsampling m windows per drive (drives with >= 32 windows, 50 repeats, mean ± std):

| m | drives | accuracy | macro-F1 |
|---|---|---|---|
| 1 | 241 | 65.9 ± 1.8 | 52.6 ± 2.3 |
| 2 | 241 | 68.3 ± 1.7 | 53.9 ± 2.1 |
| 4 | 241 | 69.5 ± 1.2 | 54.1 ± 1.9 |
| 8 | 241 | 70.4 ± 1.1 | 54.6 ± 1.7 |
| 16 | 241 | 70.9 ± 0.9 | 55.0 ± 1.5 |
| 32 | 241 | 71.1 ± 0.7 | 54.9 ± 1.0 |
| all | 241 | 71.8 ± 0.0 | 56.0 ± 0.0 |

### (e) Upper bounds

Per-car-model majority oracle: every drive of a car model receives the model's majority drive prediction (the ceiling of a predictor that is perfectly consistent within a car model).

| model | drive_full | majority oracle | seeds |
|---|---|---|---|
| feat_logreg | 48.4 | 52.5 | 1 |
| feat_hgb | 54.7 | 54.2 | 1 |
| feat_llm_logreg | 45.4 | 38.7 | 1 |
| feat_llm_hgb | 63.2 | 77.1 | 1 |
| feat_llmstats_hgb | 64.9 | 74.0 | 1 |
| resnet | 61.1 | 64.1 | 3 |
| gru | 62.9 | 72.4 | 3 |
| transformer | 61.8 | 76.3 | 3 |
| mamba2 | 61.2 | 64.0 | 3 |
| reference | 63.2 | 72.0 | - |

Footprint-noise oracle (size task): the class cut from the true footprint plus per-drive noise sigma and per-car-model bias tau (m^2, 200 repeats), drive macro-F1:

|  | tau 0.00 | tau 0.05 | tau 0.10 | tau 0.15 | tau 0.20 | tau 0.30 |
|---|---|---|---|---|---|---|
| sigma 0.00 | 100.0 | 97.4 | 92.2 | 87.3 | 80.5 | 69.8 |
| sigma 0.05 | 97.1 | 95.2 | 90.9 | 85.5 | 80.1 | 70.9 |
| sigma 0.10 | 92.1 | 90.8 | 87.4 | 83.2 | 78.0 | 68.0 |
| sigma 0.15 | 86.6 | 85.5 | 83.0 | 79.6 | 75.3 | 67.5 |
| sigma 0.20 | 80.7 | 80.0 | 78.4 | 74.9 | 72.5 | 65.8 |
| sigma 0.30 | 70.7 | 70.2 | 69.2 | 67.7 | 66.0 | 61.7 |

Curb-mass oracle (size task): two mass cuts fitted on 6364 training drives (117 car models; training macro-F1 81.7), applied to 1139 test drives with a known mass (2 without): drive macro-F1 63.3, accuracy 69.4.

Separability of the 41 held-out car models (size task): nearest differently-labelled training car model by footprint; 21 of 41 lie within 0.15 m^2 of one. Accuracy = share of the model's drives the reference predicts correctly.

| held-out car model | class | footprint | drives | accuracy | majority correct | nearest other-class training model | its class | gap m^2 |
|---|---|---|---|---|---|---|---|---|
| Hyundai Santa Cruz | LARGE | 4.94 | 6 | 0.0 | no | Ford Bronco | MID | 0.08 |
| Genesis G80 | LARGE | 4.95 | 5 | 0.0 | no | Ford Bronco | MID | 0.08 |
| Genesis GV80 | LARGE | 4.96 | 37 | 29.7 | no | Ford Bronco | MID | 0.10 |
| Tesla Model S | LARGE | 4.97 | 18 | 5.6 | no | Ford Bronco | MID | 0.11 |
| Volkswagen Atlas | LARGE | 5.11 | 20 | 0.0 | no | Ford Bronco | MID | 0.24 |
| GMC Yukon | LARGE | 5.14 | 12 | 100.0 | yes | Ford Bronco | MID | 0.28 |
| Genesis G90 | LARGE | 5.18 | 9 | 0.0 | no | Ford Bronco | MID | 0.32 |
| Toyota Sienna | LARGE | 5.21 | 11 | 36.4 | no | Ford Bronco | MID | 0.35 |
| Kia EV9 | LARGE | 5.30 | 30 | 3.3 | no | Ford Bronco | MID | 0.43 |
| Ford Ranger | LARGE | 5.30 | 10 | 100.0 | yes | Ford Bronco | MID | 0.44 |
| Kia Carnival | LARGE | 5.30 | 71 | 52.1 | yes | Ford Bronco | MID | 0.44 |
| Ford F-150 Lightning | LARGE | 6.41 | 22 | 95.5 | yes | Ford Bronco | MID | 1.55 |
| GWM Haval H6 | MID | 4.48 | 17 | 0.0 | no | Kia Sportage | SMALL | 0.06 |
| Toyota Avalon | MID | 4.50 | 4 | 50.0 | no | Kia Sportage | SMALL | 0.08 |
| Kia Stinger | MID | 4.51 | 12 | 25.0 | no | Kia Sportage | SMALL | 0.09 |
| Kia Optima | MID | 4.53 | 13 | 69.2 | yes | Kia Sportage | SMALL | 0.11 |
| Audi Q5 | MID | 4.53 | 9 | 0.0 | no | Kia Sportage | SMALL | 0.11 |
| Lexus RX | MID | 4.55 | 12 | 0.0 | no | Kia Sportage | SMALL | 0.13 |
| Nissan Altima | MID | 4.56 | 6 | 0.0 | no | Kia Sportage | SMALL | 0.14 |
| Hyundai Santa Fe | MID | 4.57 | 35 | 80.0 | yes | Kia Sportage | SMALL | 0.14 |
| Hyundai Azera | MID | 4.57 | 5 | 100.0 | yes | Kia Sportage | SMALL | 0.15 |
| Genesis GV60 | MID | 4.73 | 69 | 100.0 | yes | Hyundai Genesis | LARGE | 0.16 |
| Genesis GV70 | MID | 4.73 | 24 | 87.5 | yes | Hyundai Genesis | LARGE | 0.16 |
| Tesla Model Y | MID | 4.73 | 58 | 77.6 | yes | Hyundai Genesis | LARGE | 0.17 |
| Kia K5 | MID | 4.61 | 4 | 50.0 | yes | Kia Sportage | SMALL | 0.19 |
| Kia Sorento | MID | 4.64 | 21 | 14.3 | no | Kia Sportage | SMALL | 0.22 |
| Volkswagen ID.3 | SMALL | 4.31 | 6 | 100.0 | yes | Lexus IS | MID | 0.01 |
| Chevrolet Equinox | SMALL | 4.30 | 12 | 25.0 | no | Lexus IS | MID | 0.02 |
| Volkswagen Passat | SMALL | 4.42 | 48 | 85.4 | yes | Acura TLX | MID | 0.02 |
| Jeep Grand Cherokee | SMALL | 4.41 | 18 | 11.1 | no | Acura TLX | MID | 0.03 |
| Honda Clarity | SMALL | 4.36 | 29 | 89.7 | yes | Lexus IS | MID | 0.04 |
| Volkswagen ID.4 | SMALL | 4.36 | 17 | 100.0 | yes | Lexus IS | MID | 0.04 |
| Hyundai Ioniq | SMALL | 4.22 | 35 | 85.7 | yes | Lexus IS | MID | 0.10 |
| Volkswagen Jetta | SMALL | 4.17 | 28 | 89.3 | yes | Lexus IS | MID | 0.15 |
| Honda HR-V | SMALL | 4.17 | 24 | 58.3 | yes | Lexus IS | MID | 0.15 |
| Nissan Leaf | SMALL | 4.16 | 56 | 75.0 | yes | Lexus IS | MID | 0.16 |
| Kia Ceed | SMALL | 4.16 | 21 | 90.5 | yes | Lexus IS | MID | 0.16 |
| Chevrolet Trailblazer | SMALL | 4.10 | 8 | 87.5 | yes | Lexus IS | MID | 0.22 |
| Toyota Corolla | SMALL | 4.09 | 164 | 99.4 | yes | Lexus IS | MID | 0.22 |
| Hyundai Kona | SMALL | 4.02 | 125 | 88.8 | yes | Lexus IS | MID | 0.29 |
| Toyota Yaris | SMALL | 3.78 | 10 | 80.0 | yes | Lexus IS | MID | 0.54 |

## bev_ice / split_platform

17 saved runs (9 model families), 18295 test windows, 836 test drives with at least one window. Reference predictions for (d) and (e): mean probability of 12 deep runs.

### (a) Aggregation rule x window mask (drive macro-F1, mean over seeds)

| model | mean | logodds | median | confidence | trimmed | vote | seeds |
|---|---|---|---|---|---|---|---|
| feat_logreg | 75.9 | 76.2 | 74.5 | 75.9 | 74.8 | 74.0 | 1 |
| feat_hgb | 73.2 | 73.5 | 72.6 | 73.6 | 73.1 | 72.1 | 1 |
| feat_llm_logreg | 87.5 | 88.2 | 76.6 | 88.9 | 83.8 | 76.5 | 1 |
| feat_llm_hgb | 89.4 | 90.8 | 83.6 | 90.5 | 88.3 | 82.4 | 1 |
| feat_llmstats_hgb | 80.9 | 81.6 | 79.4 | 80.7 | 80.8 | 79.2 | 1 |
| resnet | 91.1 | 91.2 | 90.8 | 91.2 | 91.1 | 90.1 | 3 |
| gru | 89.0 | 89.6 | 88.8 | 89.5 | 89.0 | 88.8 | 3 |
| transformer | 88.4 | 88.5 | 87.9 | 88.7 | 88.4 | 87.6 | 3 |
| mamba2 | 89.4 | 89.4 | 88.7 | 89.4 | 89.3 | 88.3 | 3 |

Mean-probability rule under the window masks (drive macro-F1 (drives kept)): `all` = every test_full window; `moving` = moving_frac >= 0.5; `episodes` = n_accel_ep + n_decel_ep >= 1; `human` = human_only_strict.

| model | all | moving | episodes | human |
|---|---|---|---|---|
| feat_logreg | 75.9 (836) | 75.7 (828) | 76.7 (832) | 72.7 (564) |
| feat_hgb | 73.2 (836) | 74.3 (828) | 75.0 (832) | 74.3 (564) |
| feat_llm_logreg | 87.5 (836) | 88.0 (828) | 87.5 (832) | 85.5 (564) |
| feat_llm_hgb | 89.4 (836) | 90.0 (828) | 89.6 (832) | 89.1 (564) |
| feat_llmstats_hgb | 80.9 (836) | 81.2 (828) | 82.6 (832) | 82.6 (564) |
| resnet | 91.1 (836) | 91.1 (828) | 92.9 (832) | 89.5 (564) |
| gru | 89.0 (836) | 89.6 (828) | 91.0 (832) | 90.8 (564) |
| transformer | 88.4 (836) | 88.2 (828) | 90.3 (832) | 90.1 (564) |
| mamba2 | 89.4 (836) | 89.4 (828) | 91.8 (832) | 90.9 (564) |

### (b) Ensembles (macro-F1)

| ensemble | members | window | drive (mean prob) | drive (vote over members) | drive (stacked) | members: mean drive | members: best drive |
|---|---|---|---|---|---|---|---|
| seeds: resnet | 3 | 79.4 | 91.2 | 91.5 | 90.2 | 91.1 | 91.8 |
| seeds: gru | 3 | 80.0 | 90.3 | 88.8 | 90.2 | 89.0 | 89.1 |
| seeds: transformer | 3 | 78.0 | 87.6 | 88.3 | 87.4 | 88.4 | 89.1 |
| seeds: mamba2 | 3 | 79.0 | 89.4 | 89.5 | 89.0 | 89.4 | 89.8 |
| architectures: deep, seed 0 | 4 | 79.7 | 90.2 | 89.4 | 88.7 | 89.4 | 91.8 |
| architectures: deep, all seeds | 12 | 80.5 | 89.8 | 90.0 | 88.5 | 89.5 | 91.8 |
| deep (all seeds) + feat_hgb | 13 | 78.2 | 85.6 | 89.7 | - | 88.2 | 91.8 |
| deep (all seeds) + feat_llmstats_hgb | 13 | 78.8 | 87.0 | 89.8 | - | 88.8 | 91.8 |
| all saved runs | 17 | 80.9 | 89.9 | 90.3 | - | 87.1 | 91.8 |

### (c) Vehicle pooling

| level | class / cell | drives | vehicles | single-drive vehicles | share single-drive (%) |
|---|---|---|---|---|---|
| class | BEV | 327 | 93 | 42 | 45.2 |
| class | ICE | 509 | 151 | 67 | 44.4 |
| cell | LARGE | BEV | 70 | 20 | 10 | 50.0 |
| cell | LARGE | ICE | 164 | 48 | 18 | 37.5 |
| cell | MID | BEV | 127 | 29 | 15 | 51.7 |
| cell | MID | ICE | 110 | 43 | 21 | 48.8 |
| cell | SMALL | BEV | 130 | 44 | 17 | 38.6 |
| cell | SMALL | ICE | 233 | 58 | 26 | 44.8 |
| cell | unknown | ICE | 2 | 2 | 2 | 100.0 |
| all | all | 836 | 244 | 109 | 44.7 |

| model | drive_full | vehicle_full (vehicles) | vehicle_full_multi (vehicles) | seeds |
|---|---|---|---|---|
| feat_logreg | 75.9 | 72.7 (244) | 72.5 (135) | 1 |
| feat_hgb | 73.2 | 76.0 (244) | 72.9 (135) | 1 |
| feat_llm_logreg | 87.5 | 88.7 (244) | 90.1 (135) | 1 |
| feat_llm_hgb | 89.4 | 90.1 (244) | 92.7 (135) | 1 |
| feat_llmstats_hgb | 80.9 | 83.4 (244) | 80.6 (135) | 1 |
| resnet | 91.1 | 90.3 (244) | 92.4 (135) | 3 |
| gru | 89.0 | 88.6 (244) | 89.6 (135) | 3 |
| transformer | 88.4 | 89.0 (244) | 90.0 (135) | 3 |
| mamba2 | 89.4 | 90.6 (244) | 91.8 (135) | 3 |

### (d) Evidence sufficiency (reference predictions)

Drive accuracy with a Wilson 95 % interval by the amount of evidence in the drive (`n_windows` = selected windows; the other quantities are window-index counts summed over the drive's windows).

`n_windows`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 1-2 | 44 | 93.2 | 81.8-97.7 | 92.3 |
| 3-4 | 80 | 92.5 | 84.6-96.5 | 92.5 |
| 5-8 | 146 | 89.7 | 83.7-93.7 | 89.2 |
| 9-16 | 214 | 90.2 | 85.5-93.5 | 89.8 |
| 17-32 | 188 | 90.4 | 85.4-93.9 | 89.0 |
| 33-64 | 115 | 91.3 | 84.7-95.2 | 90.8 |
| >64 | 49 | 85.7 | 73.3-92.9 | 83.5 |

`n_accel_ep`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 6 | 66.7 | 30.0-90.3 | 62.5 |
| 1-2 | 22 | 90.9 | 72.2-97.5 | 88.5 |
| 3-4 | 43 | 90.7 | 78.4-96.3 | 90.3 |
| 5-8 | 110 | 90.0 | 83.0-94.3 | 89.8 |
| 9-16 | 199 | 92.0 | 87.3-95.0 | 91.4 |
| 17-32 | 259 | 90.7 | 86.6-93.7 | 90.2 |
| 33-64 | 151 | 88.1 | 81.9-92.3 | 87.3 |
| >64 | 46 | 93.5 | 82.5-97.8 | 91.8 |

`n_decel_ep`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 4 | 50.0 | 15.0-85.0 | 50.0 |
| 1-2 | 28 | 89.3 | 72.8-96.3 | 86.3 |
| 3-4 | 52 | 86.5 | 74.7-93.3 | 85.9 |
| 5-8 | 118 | 89.0 | 82.1-93.4 | 88.4 |
| 9-16 | 202 | 92.1 | 87.5-95.1 | 92.0 |
| 17-32 | 241 | 92.1 | 88.0-94.9 | 91.5 |
| 33-64 | 151 | 90.1 | 84.3-93.9 | 89.2 |
| >64 | 40 | 87.5 | 73.9-94.5 | 82.7 |

`n_lead_accel_events`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 584 | 88.9 | 86.1-91.2 | 88.3 |
| 1-2 | 210 | 94.3 | 90.3-96.7 | 93.6 |
| 3-4 | 28 | 89.3 | 72.8-96.3 | 88.6 |
| 5-8 | 12 | 100.0 | 75.7-100.0 | 100.0 |
| 9-16 | 2 | 100.0 | 34.2-100.0 | 100.0 |

Subsampling m windows per drive (drives with >= 32 windows, 50 repeats, mean ± std):

| m | drives | accuracy | macro-F1 |
|---|---|---|---|
| 1 | 171 | 81.1 ± 2.1 | 79.7 ± 2.3 |
| 2 | 171 | 85.2 ± 2.7 | 84.0 ± 2.9 |
| 4 | 171 | 87.5 ± 2.1 | 86.5 ± 2.2 |
| 8 | 171 | 89.4 ± 1.6 | 88.4 ± 1.7 |
| 16 | 171 | 90.2 ± 1.0 | 89.2 ± 1.1 |
| 32 | 171 | 90.2 ± 0.7 | 89.3 ± 0.8 |
| all | 171 | 90.1 ± 0.0 | 89.1 ± 0.0 |

### (e) Upper bounds

Per-car-model majority oracle: every drive of a car model receives the model's majority drive prediction (the ceiling of a predictor that is perfectly consistent within a car model).

| model | drive_full | majority oracle | seeds |
|---|---|---|---|
| feat_logreg | 75.9 | 75.8 | 1 |
| feat_hgb | 73.2 | 70.6 | 1 |
| feat_llm_logreg | 87.5 | 92.4 | 1 |
| feat_llm_hgb | 89.4 | 94.7 | 1 |
| feat_llmstats_hgb | 80.9 | 86.2 | 1 |
| resnet | 91.1 | 94.0 | 3 |
| gru | 89.0 | 92.6 | 3 |
| transformer | 88.4 | 93.4 | 3 |
| mamba2 | 89.4 | 93.0 | 3 |
| reference | 89.8 | 93.5 | - |

## size / split_device

15 saved runs (9 model families), 29214 test windows, 1218 test drives with at least one window. Reference predictions for (d) and (e): mean probability of 10 deep runs.

### (a) Aggregation rule x window mask (drive macro-F1, mean over seeds)

| model | mean | logodds | median | confidence | trimmed | vote | seeds |
|---|---|---|---|---|---|---|---|
| feat_logreg | 64.1 | 64.5 | 63.2 | 63.9 | 63.8 | 63.4 | 1 |
| feat_hgb | 76.8 | 76.4 | 76.3 | 76.6 | 76.6 | 76.2 | 1 |
| feat_llm_logreg | 53.9 | 54.0 | 46.4 | 53.9 | 51.0 | 43.1 | 1 |
| feat_llm_hgb | 69.9 | 71.0 | 63.4 | 70.9 | 67.7 | 61.6 | 1 |
| feat_llmstats_hgb | 80.6 | 80.6 | 80.5 | 80.4 | 80.3 | 79.6 | 1 |
| resnet | 77.4 | 77.6 | 77.0 | 77.3 | 77.2 | 76.8 | 3 |
| gru | 78.1 | 78.1 | 78.1 | 78.1 | 78.2 | 78.0 | 3 |
| transformer | 78.0 | 78.4 | 77.6 | 78.0 | 77.9 | 77.3 | 3 |
| mamba2 | 77.8 | 78.0 | 77.5 | 77.9 | 78.1 | 78.3 | 1 |

Mean-probability rule under the window masks (drive macro-F1 (drives kept)): `all` = every test_full window; `moving` = moving_frac >= 0.5; `episodes` = n_accel_ep + n_decel_ep >= 1; `human` = human_only_strict.

| model | all | moving | episodes | human |
|---|---|---|---|---|
| feat_logreg | 64.1 (1218) | 63.9 (1214) | 63.9 (1216) | 57.4 (804) |
| feat_hgb | 76.8 (1218) | 76.5 (1214) | 76.3 (1216) | 70.1 (804) |
| feat_llm_logreg | 53.9 (1218) | 54.0 (1214) | 54.8 (1216) | 51.7 (804) |
| feat_llm_hgb | 69.9 (1218) | 70.6 (1214) | 71.9 (1216) | 69.1 (804) |
| feat_llmstats_hgb | 80.6 (1218) | 80.5 (1214) | 80.5 (1216) | 76.7 (804) |
| resnet | 77.4 (1218) | 77.3 (1214) | 78.3 (1216) | 72.5 (804) |
| gru | 78.1 (1218) | 78.2 (1214) | 78.2 (1216) | 72.2 (804) |
| transformer | 78.0 (1218) | 77.7 (1214) | 78.5 (1216) | 73.0 (804) |
| mamba2 | 77.8 (1218) | 77.7 (1214) | 78.3 (1216) | 72.6 (804) |

### (b) Ensembles (macro-F1)

| ensemble | members | window | drive (mean prob) | drive (vote over members) | drive (stacked) | members: mean drive | members: best drive |
|---|---|---|---|---|---|---|---|
| seeds: resnet | 3 | 71.2 | 79.0 | 78.7 | 78.2 | 77.4 | 80.2 |
| seeds: gru | 3 | 71.5 | 78.3 | 78.8 | 77.5 | 78.1 | 79.2 |
| seeds: transformer | 3 | 70.6 | 78.6 | 78.9 | 77.7 | 78.0 | 78.2 |
| architectures: deep, seed 0 | 4 | 72.4 | 79.3 | 79.4 | 79.0 | 78.7 | 80.2 |
| architectures: deep, all seeds | 10 | 72.3 | 79.4 | 79.0 | 78.1 | 77.8 | 80.2 |
| deep (all seeds) + feat_hgb | 11 | 74.6 | 79.1 | 79.4 | - | 77.7 | 80.2 |
| deep (all seeds) + feat_llmstats_hgb | 11 | 74.7 | 80.9 | 79.5 | - | 78.1 | 80.6 |
| all saved runs | 15 | 73.2 | 79.7 | 80.0 | - | 74.9 | 80.6 |

### (c) Vehicle pooling

| level | class / cell | drives | vehicles | single-drive vehicles | share single-drive (%) |
|---|---|---|---|---|---|
| class | LARGE | 274 | 49 | 10 | 20.4 |
| class | MID | 353 | 67 | 17 | 25.4 |
| class | SMALL | 591 | 103 | 26 | 25.2 |
| cell | LARGE | BEV | 127 | 13 | 2 | 15.4 |
| cell | LARGE | HEV | 3 | 3 | 3 | 100.0 |
| cell | LARGE | ICE | 128 | 25 | 2 | 8.0 |
| cell | LARGE | PHEV | 2 | 2 | 2 | 100.0 |
| cell | LARGE | unknown | 14 | 6 | 1 | 16.7 |
| cell | MID | BEV | 174 | 25 | 6 | 24.0 |
| cell | MID | HEV | 41 | 11 | 2 | 18.2 |
| cell | MID | ICE | 110 | 20 | 5 | 25.0 |
| cell | MID | unknown | 28 | 14 | 6 | 42.9 |
| cell | SMALL | BEV | 93 | 17 | 2 | 11.8 |
| cell | SMALL | HEV | 86 | 20 | 9 | 45.0 |
| cell | SMALL | ICE | 289 | 42 | 10 | 23.8 |
| cell | SMALL | PHEV | 55 | 11 | 2 | 18.2 |
| cell | SMALL | unknown | 68 | 15 | 3 | 20.0 |
| all | all | 1218 | 219 | 53 | 24.2 |

| model | drive_full | vehicle_full (vehicles) | vehicle_full_multi (vehicles) | seeds |
|---|---|---|---|---|
| feat_logreg | 64.1 | 54.0 (219) | 54.9 (166) | 1 |
| feat_hgb | 76.8 | 73.6 (219) | 73.4 (166) | 1 |
| feat_llm_logreg | 53.9 | 50.8 (219) | 52.6 (166) | 1 |
| feat_llm_hgb | 69.9 | 68.9 (219) | 70.3 (166) | 1 |
| feat_llmstats_hgb | 80.6 | 78.9 (219) | 78.8 (166) | 1 |
| resnet | 77.4 | 73.9 (219) | 73.8 (166) | 3 |
| gru | 78.1 | 74.5 (219) | 75.1 (166) | 3 |
| transformer | 78.0 | 75.8 (219) | 74.4 (166) | 3 |
| mamba2 | 77.8 | 74.9 (219) | 73.8 (166) | 1 |

### (d) Evidence sufficiency (reference predictions)

Drive accuracy with a Wilson 95 % interval by the amount of evidence in the drive (`n_windows` = selected windows; the other quantities are window-index counts summed over the drive's windows).

`n_windows`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 1-2 | 54 | 66.7 | 53.4-77.8 | 59.9 |
| 3-4 | 98 | 79.6 | 70.6-86.4 | 79.4 |
| 5-8 | 217 | 86.6 | 81.5-90.5 | 85.5 |
| 9-16 | 328 | 82.6 | 78.1-86.3 | 82.2 |
| 17-32 | 279 | 79.2 | 74.1-83.6 | 78.6 |
| 33-64 | 148 | 77.7 | 70.3-83.7 | 69.4 |
| >64 | 94 | 77.7 | 68.2-84.9 | 76.7 |

`n_accel_ep`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 8 | 50.0 | 21.5-78.5 | 36.4 |
| 1-2 | 34 | 67.6 | 50.8-80.9 | 63.3 |
| 3-4 | 65 | 83.1 | 72.2-90.3 | 81.2 |
| 5-8 | 159 | 85.5 | 79.2-90.2 | 84.5 |
| 9-16 | 354 | 81.9 | 77.6-85.6 | 80.9 |
| 17-32 | 364 | 81.0 | 76.7-84.7 | 80.9 |
| 33-64 | 169 | 78.1 | 71.3-83.7 | 73.3 |
| >64 | 65 | 73.8 | 62.0-83.0 | 73.1 |

`n_decel_ep`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 5 | 80.0 | 37.6-96.4 | 92.9 |
| 1-2 | 43 | 62.8 | 47.9-75.6 | 55.0 |
| 3-4 | 69 | 73.9 | 62.5-82.8 | 72.6 |
| 5-8 | 175 | 84.0 | 77.8-88.7 | 81.9 |
| 9-16 | 359 | 83.8 | 79.7-87.3 | 83.3 |
| 17-32 | 338 | 81.7 | 77.2-85.4 | 81.2 |
| 33-64 | 169 | 77.5 | 70.6-83.2 | 73.8 |
| >64 | 60 | 75.0 | 62.8-84.2 | 74.4 |

`n_lead_accel_events`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 859 | 80.9 | 78.1-83.4 | 79.2 |
| 1-2 | 292 | 81.2 | 76.3-85.2 | 81.0 |
| 3-4 | 41 | 75.6 | 60.7-86.2 | 75.9 |
| 5-8 | 20 | 70.0 | 48.1-85.5 | 68.5 |
| 9-16 | 5 | 80.0 | 37.6-96.4 | 77.8 |
| 17-32 | 1 | 100.0 | 20.7-100.0 | 100.0 |

Subsampling m windows per drive (drives with >= 32 windows, 50 repeats, mean ± std):

| m | drives | accuracy | macro-F1 |
|---|---|---|---|
| 1 | 258 | 73.0 ± 1.5 | 69.1 ± 1.7 |
| 2 | 258 | 75.2 ± 1.5 | 71.0 ± 1.8 |
| 4 | 258 | 76.1 ± 1.1 | 71.9 ± 1.4 |
| 8 | 258 | 76.6 ± 0.9 | 72.5 ± 1.1 |
| 16 | 258 | 76.9 ± 0.7 | 73.0 ± 0.9 |
| 32 | 258 | 77.5 ± 0.5 | 73.6 ± 0.7 |
| all | 258 | 77.9 ± 0.0 | 74.1 ± 0.0 |

### (e) Upper bounds

Per-car-model majority oracle: every drive of a car model receives the model's majority drive prediction (the ceiling of a predictor that is perfectly consistent within a car model).

| model | drive_full | majority oracle | seeds |
|---|---|---|---|
| feat_logreg | 64.1 | 67.2 | 1 |
| feat_hgb | 76.8 | 80.2 | 1 |
| feat_llm_logreg | 53.9 | 60.5 | 1 |
| feat_llm_hgb | 69.9 | 80.1 | 1 |
| feat_llmstats_hgb | 80.6 | 85.1 | 1 |
| resnet | 77.4 | 82.1 | 3 |
| gru | 78.1 | 82.9 | 3 |
| transformer | 78.0 | 82.6 | 3 |
| mamba2 | 77.8 | 82.3 | 1 |
| reference | 79.4 | 84.1 | - |

Footprint-noise oracle (size task): the class cut from the true footprint plus per-drive noise sigma and per-car-model bias tau (m^2, 200 repeats), drive macro-F1:

|  | tau 0.00 | tau 0.05 | tau 0.10 | tau 0.15 | tau 0.20 | tau 0.30 |
|---|---|---|---|---|---|---|
| sigma 0.00 | 100.0 | 95.7 | 89.9 | 84.0 | 78.6 | 70.2 |
| sigma 0.05 | 95.6 | 93.1 | 88.5 | 83.5 | 79.2 | 70.2 |
| sigma 0.10 | 89.8 | 88.4 | 85.1 | 81.0 | 76.7 | 69.0 |
| sigma 0.15 | 84.4 | 83.5 | 81.2 | 78.2 | 74.4 | 67.8 |
| sigma 0.20 | 79.1 | 78.7 | 76.9 | 74.3 | 71.9 | 65.9 |
| sigma 0.30 | 70.5 | 70.3 | 69.2 | 68.3 | 66.4 | 62.2 |

Curb-mass oracle (size task): two mass cuts fitted on 6426 training drives (150 car models; training macro-F1 82.4), applied to 1218 test drives with a known mass (0 without): drive macro-F1 68.3, accuracy 70.1.

## bev_ice / split_device

15 saved runs (9 model families), 21450 test windows, 922 test drives with at least one window. Reference predictions for (d) and (e): mean probability of 10 deep runs.

### (a) Aggregation rule x window mask (drive macro-F1, mean over seeds)

| model | mean | logodds | median | confidence | trimmed | vote | seeds |
|---|---|---|---|---|---|---|---|
| feat_logreg | 85.3 | 85.5 | 84.4 | 85.7 | 84.5 | 84.1 | 1 |
| feat_hgb | 90.7 | 91.4 | 90.2 | 91.6 | 90.1 | 90.1 | 1 |
| feat_llm_logreg | 91.7 | 92.2 | 80.5 | 93.3 | 87.5 | 80.3 | 1 |
| feat_llm_hgb | 95.4 | 96.0 | 91.6 | 96.0 | 94.9 | 91.1 | 1 |
| feat_llmstats_hgb | 94.1 | 95.4 | 93.1 | 94.7 | 93.7 | 92.8 | 1 |
| resnet | 96.2 | 96.5 | 95.8 | 96.3 | 96.1 | 95.6 | 3 |
| gru | 96.5 | 96.7 | 96.1 | 96.7 | 96.5 | 95.8 | 3 |
| transformer | 95.2 | 95.6 | 94.9 | 95.4 | 95.2 | 94.8 | 3 |
| mamba2 | 96.7 | 96.8 | 96.7 | 96.8 | 96.7 | 96.4 | 1 |

Mean-probability rule under the window masks (drive macro-F1 (drives kept)): `all` = every test_full window; `moving` = moving_frac >= 0.5; `episodes` = n_accel_ep + n_decel_ep >= 1; `human` = human_only_strict.

| model | all | moving | episodes | human |
|---|---|---|---|---|
| feat_logreg | 85.3 (922) | 85.3 (919) | 87.2 (921) | 85.6 (632) |
| feat_hgb | 90.7 (922) | 90.2 (919) | 93.5 (921) | 93.1 (632) |
| feat_llm_logreg | 91.7 (922) | 91.6 (919) | 92.4 (921) | 90.4 (632) |
| feat_llm_hgb | 95.4 (922) | 95.5 (919) | 95.4 (921) | 94.2 (632) |
| feat_llmstats_hgb | 94.1 (922) | 93.9 (919) | 96.7 (921) | 95.7 (632) |
| resnet | 96.2 (922) | 96.1 (919) | 98.1 (921) | 96.9 (632) |
| gru | 96.5 (922) | 96.3 (919) | 98.7 (921) | 97.9 (632) |
| transformer | 95.2 (922) | 95.1 (919) | 97.6 (921) | 97.3 (632) |
| mamba2 | 96.7 (922) | 96.8 (919) | 98.4 (921) | 97.4 (632) |

### (b) Ensembles (macro-F1)

| ensemble | members | window | drive (mean prob) | drive (vote over members) | drive (stacked) | members: mean drive | members: best drive |
|---|---|---|---|---|---|---|---|
| seeds: resnet | 3 | 83.4 | 96.7 | 96.7 | 96.9 | 96.2 | 96.7 |
| seeds: gru | 3 | 88.3 | 96.7 | 96.8 | 97.0 | 96.5 | 96.7 |
| seeds: transformer | 3 | 87.2 | 95.6 | 95.8 | 95.8 | 95.2 | 95.9 |
| architectures: deep, seed 0 | 4 | 88.2 | 97.0 | 97.0 | 96.3 | 96.3 | 96.7 |
| architectures: deep, all seeds | 10 | 88.4 | 96.9 | 97.1 | 96.3 | 96.0 | 96.7 |
| deep (all seeds) + feat_hgb | 11 | 88.1 | 95.9 | 97.1 | - | 95.5 | 96.7 |
| deep (all seeds) + feat_llmstats_hgb | 11 | 88.3 | 96.8 | 97.2 | - | 95.8 | 96.7 |
| all saved runs | 15 | 88.7 | 97.3 | 97.4 | - | 94.5 | 96.7 |

### (c) Vehicle pooling

| level | class / cell | drives | vehicles | single-drive vehicles | share single-drive (%) |
|---|---|---|---|---|---|
| class | BEV | 394 | 55 | 10 | 18.2 |
| class | ICE | 528 | 88 | 18 | 20.5 |
| cell | LARGE | BEV | 127 | 13 | 2 | 15.4 |
| cell | LARGE | ICE | 128 | 25 | 2 | 8.0 |
| cell | MID | BEV | 174 | 25 | 6 | 24.0 |
| cell | MID | ICE | 110 | 20 | 5 | 25.0 |
| cell | SMALL | BEV | 93 | 17 | 2 | 11.8 |
| cell | SMALL | ICE | 289 | 42 | 10 | 23.8 |
| cell | unknown | ICE | 1 | 1 | 1 | 100.0 |
| all | all | 922 | 143 | 28 | 19.6 |

| model | drive_full | vehicle_full (vehicles) | vehicle_full_multi (vehicles) | seeds |
|---|---|---|---|---|
| feat_logreg | 85.3 | 87.6 (143) | 85.7 (115) | 1 |
| feat_hgb | 90.7 | 92.6 (143) | 91.8 (115) | 1 |
| feat_llm_logreg | 91.7 | 92.4 (143) | 93.4 (115) | 1 |
| feat_llm_hgb | 95.4 | 95.5 (143) | 95.4 (115) | 1 |
| feat_llmstats_hgb | 94.1 | 95.5 (143) | 94.5 (115) | 1 |
| resnet | 96.2 | 95.8 (143) | 95.7 (115) | 3 |
| gru | 96.5 | 96.0 (143) | 96.0 (115) | 3 |
| transformer | 95.2 | 96.3 (143) | 96.3 (115) | 3 |
| mamba2 | 96.7 | 96.3 (143) | 96.3 (115) | 1 |

### (d) Evidence sufficiency (reference predictions)

Drive accuracy with a Wilson 95 % interval by the amount of evidence in the drive (`n_windows` = selected windows; the other quantities are window-index counts summed over the drive's windows).

`n_windows`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 1-2 | 40 | 100.0 | 91.2-100.0 | 100.0 |
| 3-4 | 81 | 98.8 | 93.3-99.8 | 98.8 |
| 5-8 | 168 | 97.0 | 93.2-98.7 | 96.9 |
| 9-16 | 251 | 98.0 | 95.4-99.1 | 98.0 |
| 17-32 | 206 | 98.5 | 95.8-99.5 | 98.4 |
| 33-64 | 101 | 96.0 | 90.3-98.4 | 96.0 |
| >64 | 75 | 86.7 | 77.2-92.6 | 86.5 |

`n_accel_ep`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 4 | 100.0 | 51.0-100.0 | 100.0 |
| 1-2 | 24 | 100.0 | 86.2-100.0 | 100.0 |
| 3-4 | 46 | 95.7 | 85.5-98.8 | 95.5 |
| 5-8 | 126 | 93.7 | 88.0-96.7 | 93.5 |
| 9-16 | 264 | 98.5 | 96.2-99.4 | 98.4 |
| 17-32 | 283 | 97.2 | 94.5-98.6 | 97.1 |
| 33-64 | 125 | 95.2 | 89.9-97.8 | 95.0 |
| >64 | 50 | 100.0 | 92.9-100.0 | 100.0 |

`n_decel_ep`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 3 | 100.0 | 43.8-100.0 | 100.0 |
| 1-2 | 27 | 96.3 | 81.7-99.3 | 95.4 |
| 3-4 | 54 | 94.4 | 84.9-98.1 | 94.0 |
| 5-8 | 129 | 97.7 | 93.4-99.2 | 97.7 |
| 9-16 | 272 | 97.4 | 94.8-98.7 | 97.4 |
| 17-32 | 265 | 97.0 | 94.2-98.5 | 96.9 |
| 33-64 | 128 | 95.3 | 90.2-97.8 | 95.1 |
| >64 | 44 | 100.0 | 92.0-100.0 | 100.0 |

`n_lead_accel_events`

| bin | drives | accuracy | 95 % CI | macro-F1 |
|---|---|---|---|---|
| 0 | 633 | 96.5 | 94.8-97.7 | 96.4 |
| 1-2 | 236 | 98.3 | 95.7-99.3 | 98.3 |
| 3-4 | 35 | 94.3 | 81.4-98.4 | 94.3 |
| 5-8 | 13 | 100.0 | 77.2-100.0 | 100.0 |
| 9-16 | 5 | 100.0 | 56.6-100.0 | 100.0 |

Subsampling m windows per drive (drives with >= 32 windows, 50 repeats, mean ± std):

| m | drives | accuracy | macro-F1 |
|---|---|---|---|
| 1 | 190 | 85.9 ± 2.3 | 85.8 ± 2.4 |
| 2 | 190 | 90.2 ± 1.3 | 90.0 ± 1.4 |
| 4 | 190 | 91.8 ± 1.4 | 91.6 ± 1.4 |
| 8 | 190 | 92.3 ± 1.0 | 92.2 ± 1.1 |
| 16 | 190 | 92.6 ± 0.8 | 92.5 ± 0.9 |
| 32 | 190 | 92.3 ± 0.7 | 92.2 ± 0.7 |
| all | 190 | 92.6 ± 0.0 | 92.4 ± 0.0 |

### (e) Upper bounds

Per-car-model majority oracle: every drive of a car model receives the model's majority drive prediction (the ceiling of a predictor that is perfectly consistent within a car model).

| model | drive_full | majority oracle | seeds |
|---|---|---|---|
| feat_logreg | 85.3 | 90.7 | 1 |
| feat_hgb | 90.7 | 91.9 | 1 |
| feat_llm_logreg | 91.7 | 97.5 | 1 |
| feat_llm_hgb | 95.4 | 96.7 | 1 |
| feat_llmstats_hgb | 94.1 | 98.0 | 1 |
| resnet | 96.2 | 97.9 | 3 |
| gru | 96.5 | 98.3 | 3 |
| transformer | 95.2 | 98.4 | 3 |
| mamba2 | 96.7 | 98.3 | 1 |
| reference | 96.9 | 98.3 | - |
