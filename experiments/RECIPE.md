# Recipe grid on the internal model-disjoint folds

Window-model recipes of plan part II stage 2, trained and selected on the internal car-model-disjoint folds
(experiments/INTERNAL_FOLDS.md); the published validation split shares most car models with the training split, so
nothing here looks at it or at the test split. Every row is one run of `experiments.train.run` with the keyword
arguments printed by `experiments/run_recipe.py --dry-run`; `inner drive_full F1` is the macro-F1 of the held-out
fold's drives (mean class probability over the drive's windows), `inner window F1` the window-level macro-F1 of the
same fold, both at the selected epoch. The ladder is cumulative from S1 to S6; S7-S9 each add one change to S5. B0-B5
are the BEV vs ICE counterparts (no ordinal head, no cutmix: the task is binary and the LARGE-class composition problem
does not apply). L1-L4 are leave-one-car-model-out probes implemented as single-model folds (the probe car model is
held out of train and validation, every drive of an identity group that drove it is removed from the training set).
Stage `folds` runs the best recipe on all three folds with the GRU and the ResNet to produce the out-of-fold
probabilities used by experiments/aggregate.py.

<!-- rendered by experiments/run_recipe.py: everything below this line is overwritten -->

## Size: SMALL / MID / LARGE

Best recipe so far by inner drive_full macro-F1: `S9` (used by `--stage folds` unless --recipes is given).

### gru, fold 0

| id | changes | inner drive_full F1 | inner window F1 | LARGE recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| S0 | reference recipe (class-balanced sampling, cross-entropy only) | 65.3 | 61.2 | 37.4 | softmax | 4 | 1 | 1 |
| S1 | sampling by car model inside each class (sampling=class_model) | 62.8 | 61.4 | 38.8 | softmax | 1 | 1 | 1 |
| S2 | S1 + footprint regression head (aux fp) | 66.8 | 65.3 | 36.7 | softmax | 1 | 1 | 1 |
| S3 | S2 + CORAL ordinal head | 64.8 | 66.3 | 28.8 | softmax | 1 | 1 | 1 |
| S4 | S3 + curb-mass and four-class powertrain heads (mass, pt4) | 65.2 | 67.0 | 30.9 | softmax | 1 | 1 | 1 |
| S5 | S4 + physics-safe augmentation (flip, mask, chdrop, noise) | 66.2 | 64.0 | 41.7 | softmax | 5 | 1 | 1 |
| S6 | S5 + same-class cutmix | 64.8 | 64.0 | 45.3 | softmax | 5 | 2 | 1 |
| S7 | S5 + DAS down-weighting (das_beta 0.5) | 65.6 | 65.0 | 36.7 | softmax | 8 | 2 | 1 |
| S8 | S5 + conditional adversarial car-model head (adv_lambda 0.1) | 66.5 | 64.7 | 42.4 | softmax | 9 | 2 | 1 |
| S9 | S5 + group-DRO-lite | 66.9 | 61.8 | 39.6 | softmax | 11 | 2 | 1 |

### gru, fold 1

| id | changes | inner drive_full F1 | inner window F1 | LARGE recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| S0 | reference recipe (class-balanced sampling, cross-entropy only) | 69.5 | 63.8 | 88.9 | softmax | 3 | 1 | 1 |
| S2 | S1 + footprint regression head (aux fp) | 69.7 | 65.1 | 85.9 | softmax | 5 | 2 | 1 |
| S5 | S4 + physics-safe augmentation (flip, mask, chdrop, noise) | 70.1 | 64.4 | 86.3 | softmax | 7 | 3 | 1 |
| S9 | S5 + group-DRO-lite | 70.5 | 59.2 | 86.3 | softmax | 13 | 4 | 1 |

### gru, fold 2

| id | changes | inner drive_full F1 | inner window F1 | LARGE recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| S0 | reference recipe (class-balanced sampling, cross-entropy only) | 60.4 | 51.5 | 39.1 | softmax | 5 | 1 | 1 |
| S2 | S1 + footprint regression head (aux fp) | 60.0 | 50.7 | 30.1 | softmax | 2 | 1 | 1 |
| S5 | S4 + physics-safe augmentation (flip, mask, chdrop, noise) | 63.3 | 52.1 | 61.3 | softmax | 4 | 2 | 1 |
| S9 | S5 + group-DRO-lite | 64.8 | 48.9 | 73.0 | softmax | 7 | 2 | 1 |

### resnet, fold 0

| id | changes | inner drive_full F1 | inner window F1 | LARGE recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| S0 | reference recipe (class-balanced sampling, cross-entropy only) | 65.0 | 63.1 | 43.2 | softmax | 7 | 1 | 1 |
| S9 | S5 + group-DRO-lite | 67.0 | 61.0 | 39.6 | softmax | 16 | 3 | 1 |

### resnet, fold 1

| id | changes | inner drive_full F1 | inner window F1 | LARGE recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| S0 | reference recipe (class-balanced sampling, cross-entropy only) | 70.3 | 64.5 | 90.2 | softmax | 8 | 2 | 1 |
| S9 | S5 + group-DRO-lite | 70.4 | 59.6 | 87.6 | softmax | 21 | 3 | 1 |

### resnet, fold 2

| id | changes | inner drive_full F1 | inner window F1 | LARGE recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| S0 | reference recipe (class-balanced sampling, cross-entropy only) | 64.5 | 52.4 | 59.0 | softmax | 1 | 1 | 1 |
| S9 | S5 + group-DRO-lite | 61.2 | 48.1 | 62.1 | softmax | 6 | 1 | 1 |

### Leave-one-car-model-out probes

| id | held-out car model | model | inner drive_full F1 | inner window F1 | best epoch | minutes |
|---|---|---|---|---|---|---|
| L1_S9 | Honda Odyssey | gru | 74.7 | 44.4 | 11 | 3 |
| L2_S9 | Hyundai Palisade | gru | 27.6 | 44.0 | 2 | 1 |
| L3_S9 | Hyundai Sonata | gru | 18.3 | 37.4 | 2 | 1 |
| L4_S9 | Tesla Model 3 | gru | 11.3 | 27.4 | 5 | 2 |

## Powertrain: BEV vs ICE

Best recipe so far by inner drive_full macro-F1: `B5` (used by `--stage folds` unless --recipes is given).

### gru, fold 0

| id | changes | inner drive_full F1 | inner window F1 | BEV recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| B0 | reference recipe (class-balanced sampling, cross-entropy only) | 89.4 | 76.0 | 90.2 | softmax | 1 | 1 | 1 |
| B1 | sampling by car model inside each class (sampling=class_model) | 88.0 | 78.4 | 91.5 | softmax | 3 | 1 | 1 |
| B2 | B1 + footprint regression head (aux fp) | 87.4 | 77.5 | 89.6 | softmax | 3 | 1 | 1 |
| B3 | B2 + curb-mass and four-class powertrain heads (mass, pt4) | 88.7 | 82.4 | 91.5 | softmax | 6 | 1 | 1 |
| B4 | B3 + physics-safe augmentation (flip, mask, chdrop, noise) | 92.8 | 71.1 | 97.7 | softmax | 1 | 1 | 1 |
| B5 | B4 + DAS down-weighting (das_beta 0.5) | 92.8 | 72.9 | 86.6 | softmax | 1 | 1 | 1 |

### gru, fold 1

| id | changes | inner drive_full F1 | inner window F1 | BEV recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| B0 | reference recipe (class-balanced sampling, cross-entropy only) | 90.8 | 74.4 | 86.7 | softmax | 2 | 1 | 1 |
| B4 | B3 + physics-safe augmentation (flip, mask, chdrop, noise) | 90.9 | 71.5 | 82.5 | softmax | 2 | 1 | 1 |
| B5 | B4 + DAS down-weighting (das_beta 0.5) | 91.1 | 72.0 | 80.1 | softmax | 2 | 1 | 1 |

### gru, fold 2

| id | changes | inner drive_full F1 | inner window F1 | BEV recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| B0 | reference recipe (class-balanced sampling, cross-entropy only) | 95.4 | 83.7 | 93.3 | softmax | 7 | 1 | 1 |
| B4 | B3 + physics-safe augmentation (flip, mask, chdrop, noise) | 94.6 | 81.9 | 92.8 | softmax | 5 | 1 | 1 |
| B5 | B4 + DAS down-weighting (das_beta 0.5) | 95.4 | 78.2 | 92.5 | softmax | 2 | 1 | 1 |

### resnet, fold 0

| id | changes | inner drive_full F1 | inner window F1 | BEV recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| B0 | reference recipe (class-balanced sampling, cross-entropy only) | 90.0 | 71.5 | 96.1 | softmax | 2 | 1 | 1 |
| B5 | B4 + DAS down-weighting (das_beta 0.5) | 95.7 | 73.1 | 96.4 | softmax | 2 | 1 | 1 |

### resnet, fold 1

| id | changes | inner drive_full F1 | inner window F1 | BEV recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| B0 | reference recipe (class-balanced sampling, cross-entropy only) | 93.2 | 78.2 | 85.8 | softmax | 6 | 1 | 1 |
| B5 | B4 + DAS down-weighting (das_beta 0.5) | 93.9 | 77.6 | 85.8 | softmax | 13 | 2 | 1 |

### resnet, fold 2

| id | changes | inner drive_full F1 | inner window F1 | BEV recall (drive) | decision | best epoch | minutes | seeds |
|---|---|---|---|---|---|---|---|---|
| B0 | reference recipe (class-balanced sampling, cross-entropy only) | 96.0 | 81.1 | 91.2 | softmax | 7 | 1 | 1 |
| B5 | B4 + DAS down-weighting (das_beta 0.5) | 96.1 | 80.4 | 95.1 | softmax | 7 | 1 | 1 |
