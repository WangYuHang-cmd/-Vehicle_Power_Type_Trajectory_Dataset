# Pre-registration of the method-research campaign (Part II)

Written on 7 Nov 2026, before the first fold run of the campaign. Everything below binds the later stages; a deviation
is recorded in the log at the end of this page, never applied silently. Plan reference: Part II of the approved plan
(stages 0-6, gates G0-G5).

## Goal and headline quantity

- Headline: drive-level macro-F1 under `drive_full` (docs/PROTOCOLS.md, "Drive- and vehicle-level protocols") on
  held-out car models (`split_platform`, test split), for both tasks (size: SMALL / MID / LARGE; BEV vs ICE).
- Secondary, always reported next to it: `drive_human_only`, `vehicle_full` (and `vehicle_full_multi`), the window-level
  protocols `test_full` / `test_balanced` / `test_human_only`, and every number again under `split_device`.
- Every unit-level number carries the cluster-bootstrap 95 % interval over car models (1,000 resamples, seed 0).

## Selection sets (what may decide anything)

- Hyper-parameters, recipe choices, early stopping or the fixed epoch count, the aggregation rule, the decision rule and
  the membership of an ensemble are chosen on the **internal model-disjoint folds** (`experiments/internal_folds.py`:
  three folds over the train + validation pool of `split_platform`, each holding out whole car models; the fold's
  training set drops every drive whose identity group also drove a held-out model) and, where this page says so,
  confirmed on the released validation split.
- The released validation split shares car models with the training split (docs/PROTOCOLS.md), so a validation score
  rewards recognising a known car model. It is reported, and it may confirm a fold decision, but it never decides alone.
- The test split is evaluated **once per final model** by the frozen `experiments/eval_final.py` (stage 5). No number
  computed on the test split before that point may select anything.
- The stage-0 oracle studies (`experiments/ORACLES.md`) read the test split by construction. They are post-hoc ceilings
  for the already-published models; any idea they suggest has to win on the folds under the rules below before it enters
  a final model.

## Fixed settings

- Seeds 0 / 1 / 2 for every final configuration; Mamba-2 runs seed 0 only (training time).
- Final models train on the whole `train` split of the scheme with a **fixed number of epochs = median of the folds'
  best epochs**; the released validation split is only reported for them.
- **Aggregation**: the mean of the per-window probability vectors stays the default. A variant (log-odds mean, median,
  confidence-weighted mean, trimmed mean, majority vote, information weights such as 1 - DAS share or episode counts,
  a learned gate) replaces it only if it beats the mean by **>= 1.0 drive-level macro-F1 points** on the folds'
  out-of-fold predictions **and** on the released validation split, **for both tasks**. One rule for both tasks.
- **Decision rule** (softmax arg-max / footprint-regression bucket / CORAL / their average) is chosen per task on the
  folds' out-of-fold `drive_full`; ties keep softmax arg-max.
- **Ensembling, stacking or distillation** enter the final list only if the fold `drive_full` gain over the best
  single model is **>= +1.0 points** (G4); the stacker is fitted on out-of-fold predictions only.
- Driving automation: windows and episodes with DAS activity stay in training (`das_enabled` is an official input);
  DAS down-weighting is one ablation. `drive_full` and `drive_human_only` are always reported as a pair.
- Nothing changes in the released data, labels, splits or window protocols.

## Gates

| gate | stage | pass rule (fold numbers unless stated) | on failure |
|---|---|---|---|
| G0 | 0 | the new unit protocols reproduce every stored `drive_level` number (126 run x protocol pairs, 1e-9); this page exists | fix before any run |
| G1 | 1 | fold `drive_full` of the existing recipe (GRU, ResNet) within 5 points of the published test value (expected size 60-66, BEV vs ICE 85-90) | redesign the folds |
| G2 size | 2 | fold `drive_full` >= 70 | stop chasing 90 on size; report the ceilings; size effort moves to `split_device`, the vehicle level and footprint regression |
| G2 BEV vs ICE | 2 | fold `drive_full` >= 90 and >= +1 over the baseline recipe (or `drive_human_only` >= 92) | keep the baseline recipe |
| G3 | 3 | episode MIL model: fold `drive_full` >= +2 over the best window recipe (size) or >= 91 (BEV vs ICE) | MIL reported as a single-seed ablation only |
| G4 | 4 | ensemble / stack: fold `drive_full` >= +1.0 over the best single model | no ensemble in the final list |
| G5 | 5 | the final test evaluation runs once per listed model; a re-run only after a crash; when the two stopping rules (fixed epochs vs validation early stopping) differ by more than 1 point on the test split this is reported, not re-selected | - |

## Frozen evaluation

- `experiments/eval_final.py` (added in stage 5) reads `experiments/final_manifest.json` and evaluates every listed
  model once on the test split: window protocols, `drive_full`, `drive_human_only`, `vehicle_full`,
  `vehicle_full_multi`, cluster-bootstrap intervals; it writes one JSON per model and never trains.
- sha256 of the frozen `eval_final.py`: (to be filled in when the file is frozen, before the first test run)
- Final manifest sha256: (to be filled in with the previous line)

## Log

| date | entry |
|---|---|
| 7 Nov 2026 | G0 passed: `starter_kit.eval.evaluate_units` reproduces the stored `drive_level` macro-F1, confusion and drive counts of all 63 saved runs under both `drive_full` and `drive_human_only` (126 pairs, exact). Pre-registration written. |

- 7 Nov 2026, before stage 1: internal folds built with the documented **fleet rule** (`experiments/internal_folds.py`): an identity group that spans five or more car models (one such group exists: a fleet of 33 vehicles, about 480 h, all in the training split) is pinned to the training side; its drives of held-out models are dropped from the fold instead of being validated. The literal group rule would have removed 13-17 % of the pool windows from every fold's training set. Revert path: `--fleet-min-models 0`.
- Consequence recorded for G1/G2: no fold holds out a LARGE x BEV model (the cell is 95 % one model, the other candidates have one group or too few windows), so fold numbers for that cell reflect training models only; the test split does hold out LARGE x BEV models.
- Stage 0 done: G0 passed (126 run x protocol pairs reproduced exactly); ORACLES.md written (post-hoc ceilings); validation probabilities regenerated for every checkpoint so the stacking column of ORACLES.md can be filled.
- 7 Nov 2026, stage 1 done: G1 passed. Fold `drive_full` of the reference recipe (mean over the three folds): size GRU 65.1 / ResNet 66.6 against published test 62.9 / 61.1; BEV vs ICE GRU 91.9 / ResNet 93.1 against 89.0 / 91.1 (all within 5 points). Best epochs on the model-disjoint folds are 1-8 (median about 4), far below the 16-22 selected by the shared-model validation split: the published validation rewards memorising car models.
- 7 Nov 2026, stage 2 sweep (fold 0, GRU, one seed): size S0 65.3; S1 62.8; S2 66.8; S3 64.8; S4 65.2; S5 66.2; S6 64.8; S7 65.6; S8 66.5; S9 66.9 (spread of 4 points, comparable to the fold-to-fold spread; no recipe near 70). BEV vs ICE B0 89.4; B1 88.0; B2 87.4; B3 88.7; B4 92.8; B5 92.8 (augmentation helps on fold 0). Leave-one-model-out probes with S9: Odyssey 74.7, Palisade 27.6, Sonata 18.3, Model 3 11.3 drive F1 (single held-out models near the thresholds are systematically misread). Candidates S2 / S5 / S9 and B4 / B5 go to the remaining folds before the G2 decision.
- 7 Nov 2026, aggregation decision (out-of-fold, reference recipe, GRU + ResNet pooled folds): size `drive_full` mean 67.5 -> best variant 67.8 (+0.3; the +3.6 seen post hoc on the test predictions does not reproduce out of fold); BEV vs ICE mean 93.0 -> DAS-down-weighted log-odds mean 94.8 (+1.8). The pre-registered rule needs >= +1.0 for both tasks, so **the mean of the window probabilities stays the single published aggregation**; the DAS-weighted variant is reported for BEV vs ICE as a supplementary row only.
- 7 Nov 2026, G2 size (three folds, GRU, one seed, fold `drive_full`): S0 65.1 (65.3 / 69.5 / 60.4), S2 65.5, S5 66.5, S9 67.4 (66.9 / 70.5 / 64.8). Best recipe S9 (car-model-balanced sampling + footprint, CORAL, mass and powertrain-4 heads + physics-safe augmentation + group-DRO-lite) gains about +2 over the reference but stays below 70: **G2 size fails; the 90 target is not pursued further for size on held-out car models.** S9 is still carried to the final list as the best window recipe; the ceiling statement goes into the report. Size effort beyond that is limited to the vehicle-level and split_device numbers and the footprint-regression error.
- 7 Nov 2026, G2 BEV vs ICE (three folds, GRU): B0 91.9 (89.4 / 90.8 / 95.4), B4 92.8, B5 93.1 (92.8 / 91.1 / 95.4); ResNet B0 93.1. **G2 passes with B5** (car-model-balanced sampling + footprint, mass and powertrain-4 heads with hybrid windows + augmentation + DAS down-weighting): fold `drive_full` >= 90 and +1.2 over the reference. Best epochs on the folds are 1-2, so the final B5 models train for 2 epochs.
- 7 Nov 2026, G3 (episode MIL, gated attention, three folds, one seed): direct-bag `drive_full` on the folds' held-out models: size 67.8 / 65.6 / 67.5 (mean 67.0, equal to the best window recipe S9 at 67.4: **G3 size fails**, the MIL size output is reported as an ablation); BEV vs ICE 96.0 / 98.2 / 93.2 (mean 95.8, +2.7 over B5 and above 91: **G3 BEV vs ICE passes**). The window-mean aggregation of the same MIL model is weaker (size 61.1, BEV vs ICE 88.8); the model's own bag prediction is its drive-level output. Final MIL runs: 10 epochs (median of the fold best epochs 7 / 10 / 13), official validation reported only, full-split window embeddings.
- 7 Nov 2026, G4 (ensembles, out of fold): GRU + ResNet mean of probabilities 93.9 against the best single member 93.1 on BEV vs ICE (+0.8), stacked 92.3; size +0.3. Below the +1.0 rule: **no ensemble in the final list**.
- Final list (stage 5): size - GRU S9 (11 epochs) and ResNet S9 (16 epochs); BEV vs ICE - GRU B5 (2 epochs) and ResNet B5 or B0 (fixed epochs = fold median, decided when the ResNet B5 folds finish); joint episode MIL (gated attention, 10 epochs); three seeds each; test evaluated once by eval_final.py.
- 7 Nov 2026, supplementary aggregation (fixed before the final test run): besides the headline mean of window probabilities, every final model is also reported with the log-odds mean weighted by (1 - DAS share + 0.1)^0.5 x (1 + acceleration + deceleration episodes)^1 (out of fold on the final recipes: size +0.9, BEV vs ICE +1.8). It is a supplementary row, not the headline.
- 7 Nov 2026, frozen evaluator: experiments/eval_final.py sha256 = ebe7c9c2ebf9332a9285b0fa5ede52f33ddd35ff17fbf4c5a7d4bbf0cb32e0fd. The manifest experiments/final_manifest.json is written next; its sha256 is recorded when the final runs start.
- 7 Nov 2026, ResNet B5 folds: 95.7 / 93.9 / 96.1 (mean 95.2) against ResNet B0 93.1: ResNet B5 (7 epochs) joins the final list. Final manifest written: experiments/final_manifest.json sha256 = 9d28b7fbd54495ce241198da985effb000f24a95307edc3d47e8390901ea80dc (size: GRU S9 11 ep, ResNet S9 16 ep; BEV vs ICE: GRU B5 2 ep, ResNet B5 7 ep; joint episode MIL 10 ep; three seeds on split_platform, split_device with three seeds for the window models and one for the MIL). Stage 5 starts now.
- 7 Nov 2026, stage 5 done (test split evaluated once per entry by eval_final.py; split_platform, mean over three seeds, drive_full / vehicle_full / window test_full macro-F1): size GRU S9 67.1 / 60.0 / 53.2; ResNet S9 66.8 / 62.6 / 53.2; episode MIL 64.6 (own bag prediction 64.3) / 62.3 / 54.2 - against the published GRU 62.9 and ResNet 61.1 at drive level (+4 to +6 points; the window level is unchanged). BEV vs ICE: GRU B5 92.6 / 93.4 / 77.0; ResNet B5 93.0 / 92.8 / 79.9; episode MIL 90.2 (own bag prediction 93.0) / 90.4 / 78.8 - against the published ResNet 91.1 and GRU 89.0 (+2 to +4). The MIL's fold advantage on BEV vs ICE (95.8 on the folds) did not carry to the test models (93.0): reported as such. split_device: size S9 drive 79.5 (GRU) / 80.6 (ResNet), vehicle 75.8 / 76.4; BEV vs ICE B5 drive 95.1 / 97.2, vehicle 96.3 / 99.0. Correction during stage 5: the split_device MIL entry needed split_device window embeddings (provenance rule); the manifest entry was corrected and that entry run afterwards (recorded here, no other change).
- 7 Nov 2026, supplementary aggregation on the final S9 / B5 models (test, six runs each, pre-specified rule = log-odds mean with weights (1 - DAS share + 0.1)^0.5 x (1 + episodes)): size `drive_full` 66.9 -> 68.7 (+1.7, 6/6 runs improved), BEV vs ICE 92.8 -> 93.5 (+0.7, 6/6). Headline stays the plain mean as pre-registered.
- 7 Nov 2026, paper draft: the result JSONs carry no bootstrap interval (experiments.train.unit_level does not run it), so the 95 % cluster-bootstrap intervals over car models were computed post hoc from the saved test predictions by experiments/ci_table.py (starter_kit.eval.evaluate_units, 1,000 resamples, seed 0) into paper/tables/ci.json; the point estimates reproduce the stored unit_level numbers exactly. The paper folder (paper/) is local and not part of the public export.
