# Classification results (60-s windows, official input channels only)

Macro-F1 in percent; mean ± std over seeds when several seeds ran. `window` = one prediction per 60-s window, `drive` = mean class probability over the windows of a drive, `vehicle` = the same over all test drives of a vehicle. Protocols follow docs/PROTOCOLS.md: `full` = every eligible test window, `balanced` = the published class-balanced list, `human` = windows without any driving automation activity (strict). Chance level for macro-F1 is about 33 (size) and 50 (BEV vs ICE); the always-majority predictor scores below that.

## Headline: car models held out (`split_platform`)

### Size: SMALL / MID / LARGE

| model | params | window full | window balanced | window human-only | drive full | drive human-only | vehicle full | best val | seeds |
|---|---|---|---|---|---|---|---|---|---|
| hand-crafted features + logistic regression | 0 | 43.6 | 40.5 | 40.2 | 48.4 | 43.6 | - | 53.7 | 1 |
| hand-crafted features + gradient boosting | 0 | 49.7 | 45.5 | 46.3 | 54.7 | 50.5 | - | 68.6 | 1 |
| LLM physical features + logistic regression | 0 | 32.2 | 36.1 | 35.8 | 45.4 | 42.1 | - | 34.9 | 1 |
| LLM physical features + gradient boosting | 0 | 40.5 | 43.6 | 47.3 | 63.2 | 58.9 | - | 42.5 | 1 |
| LLM physical + hand-crafted features + gradient boosting | 0 | 54.1 | 49.6 | 49.8 | 64.9 | 64.2 | - | 71.6 | 1 |
| 1-D ResNet (10 Hz) | 2,309,251 | 53.6 ± 1.0 | 49.4 ± 0.3 | 48.0 ± 0.9 | 61.1 ± 1.3 | 60.3 ± 2.1 | - | 71.8 ± 0.7 | 3 |
| bi-GRU | 728,067 | 54.7 ± 0.3 | 50.7 ± 0.3 | 50.4 ± 0.2 | 62.9 ± 0.8 | 63.3 ± 1.0 | - | 71.0 ± 0.3 | 3 |
| Transformer encoder | 697,091 | 52.5 ± 0.8 | 48.5 ± 0.6 | 47.7 ± 0.3 | 61.8 ± 0.9 | 61.3 ± 0.2 | - | 70.1 ± 0.6 | 3 |
| Mamba-2 (SSD), bidirectional | 1,073,891 | 53.1 ± 0.4 | 49.6 ± 0.2 | 49.8 ± 0.7 | 61.2 ± 0.3 | 61.1 ± 0.5 | - | 69.6 ± 0.1 | 3 |
| bi-GRU, recipe S9 (model-balanced sampling, footprint/CORAL/mass/powertrain heads, augmentation, group-DRO) | 729,867 | 53.2 ± 0.3 | 50.1 ± 0.3 | 47.8 ± 0.3 | 67.1 ± 0.0 | 61.7 ± 0.3 | 60.0 ± 0.6 | 60.7 ± 1.2 | 3 |
| 1-D ResNet, recipe S9 | 2,311,051 | 53.2 ± 0.8 | 50.3 ± 0.5 | 49.5 ± 1.0 | 66.8 ± 1.7 | 62.9 ± 1.5 | 62.6 ± 1.4 | 62.8 ± 1.1 | 3 |
| episode MIL (gated attention over transients + window embeddings) | 990,676 | 54.2 ± 0.3 | 51.0 ± 0.5 | 50.0 ± 0.7 | 64.6 ± 0.5 | 62.5 ± 1.3 | 62.3 ± 0.9 | 79.1 ± 0.5 | 3 |

Best single run on `full`: bi-GRU (seed 1), per-class recall: SMALL 80.9, MID 50.9, LARGE 32.4.

### Powertrain: BEV vs ICE

| model | params | window full | window balanced | window human-only | drive full | drive human-only | vehicle full | best val | seeds |
|---|---|---|---|---|---|---|---|---|---|
| hand-crafted features + logistic regression | 0 | 67.2 | 68.7 | 67.3 | 75.9 | 72.7 | - | 64.5 | 1 |
| hand-crafted features + gradient boosting | 0 | 69.5 | 71.3 | 73.5 | 73.2 | 74.3 | - | 83.7 | 1 |
| LLM physical features + logistic regression | 0 | 67.1 | 63.9 | 76.7 | 87.5 | 85.5 | - | 63.8 | 1 |
| LLM physical features + gradient boosting | 0 | 69.6 | 71.5 | 84.2 | 89.4 | 89.1 | - | 71.3 | 1 |
| LLM physical + hand-crafted features + gradient boosting | 0 | 73.2 | 74.9 | 80.2 | 80.9 | 82.6 | - | 85.6 | 1 |
| 1-D ResNet (10 Hz) | 2,308,994 | 78.4 ± 0.4 | 78.5 ± 0.7 | 83.2 ± 0.6 | 91.1 ± 0.8 | 89.5 ± 0.7 | - | 83.4 ± 1.2 | 3 |
| bi-GRU | 727,810 | 79.2 ± 0.8 | 79.5 ± 0.7 | 85.2 ± 0.3 | 89.0 ± 0.1 | 90.8 ± 0.8 | - | 86.9 ± 0.3 | 3 |
| Transformer encoder | 696,962 | 77.0 ± 0.4 | 76.7 ± 0.5 | 83.5 ± 1.1 | 88.4 ± 0.6 | 90.1 ± 0.4 | - | 89.2 ± 0.5 | 3 |
| Mamba-2 (SSD), bidirectional | 1,073,762 | 77.3 ± 0.7 | 78.4 ± 0.8 | 84.0 ± 0.9 | 89.4 ± 0.3 | 90.9 ± 0.2 | - | 87.0 ± 0.0 | 3 |
| bi-GRU, recipe B5 (model-balanced sampling, aux heads with hybrids, augmentation, DAS down-weighting) | 729,352 | 77.0 ± 0.3 | 77.4 ± 0.2 | 83.7 ± 0.4 | 92.6 ± 0.1 | 89.1 ± 0.3 | 93.4 ± 0.4 | 75.6 ± 0.4 | 3 |
| 1-D ResNet, recipe B5 | 2,310,536 | 79.9 ± 0.7 | 80.2 ± 0.8 | 83.2 ± 0.6 | 93.0 ± 0.8 | 88.7 ± 1.0 | 92.8 ± 0.4 | 79.3 ± 0.2 | 3 |
| episode MIL (gated attention over transients + window embeddings) | 990,676 | 78.8 ± 0.1 | 78.5 ± 0.1 | 82.2 ± 0.2 | 90.2 ± 0.2 | 89.9 ± 0.2 | 90.4 ± 0.5 | 97.5 ± 0.4 | 3 |

Best single run on `full`: 1-D ResNet, recipe B5 (seed 1), per-class recall: BEV 77.0, ICE 85.6.

## Secondary: devices held out, models shared (`split_device`)

### Size: SMALL / MID / LARGE

| model | params | window full | window balanced | window human-only | drive full | drive human-only | vehicle full | best val | seeds |
|---|---|---|---|---|---|---|---|---|---|
| hand-crafted features + logistic regression | 0 | 55.8 | 55.1 | 47.8 | 64.1 | 57.4 | - | 53.2 | 1 |
| hand-crafted features + gradient boosting | 0 | 70.8 | 67.5 | 61.8 | 76.8 | 70.1 | - | 70.6 | 1 |
| LLM physical features + logistic regression | 0 | 35.3 | 38.9 | 43.8 | 53.9 | 51.7 | - | 34.3 | 1 |
| LLM physical features + gradient boosting | 0 | 44.5 | 46.7 | 57.9 | 69.9 | 69.1 | - | 40.7 | 1 |
| LLM physical + hand-crafted features + gradient boosting | 0 | 72.4 | 69.2 | 64.5 | 80.6 | 76.7 | - | 73.2 | 1 |
| 1-D ResNet (10 Hz) | 2,309,251 | 68.4 ± 3.0 | 65.6 ± 2.1 | 59.3 ± 1.2 | 77.4 ± 2.2 | 72.5 ± 1.5 | - | 70.6 ± 1.2 | 3 |
| bi-GRU | 728,067 | 70.2 ± 0.9 | 67.7 ± 0.8 | 61.1 ± 1.7 | 78.1 ± 1.4 | 72.2 ± 1.3 | - | 72.0 ± 0.7 | 3 |
| Transformer encoder | 697,091 | 68.9 ± 0.4 | 67.0 ± 0.4 | 61.3 ± 1.9 | 78.0 ± 0.2 | 73.0 ± 1.4 | - | 70.7 ± 0.5 | 3 |
| Mamba-2 (SSD), bidirectional | 1,073,891 | 69.8 | 67.7 | 60.9 | 77.8 | 72.6 | - | 71.2 | 1 |
| bi-GRU, recipe S9 (model-balanced sampling, footprint/CORAL/mass/powertrain heads, augmentation, group-DRO) | 729,867 | 62.9 ± 1.0 | 61.6 ± 1.1 | 63.1 ± 1.0 | 79.5 ± 0.7 | 75.6 ± 0.7 | 75.8 ± 0.7 | 58.4 ± 1.1 | 3 |
| 1-D ResNet, recipe S9 | 2,311,051 | 64.8 ± 0.3 | 62.8 ± 0.4 | 62.9 ± 0.7 | 80.6 ± 0.1 | 76.2 ± 1.3 | 76.4 ± 0.8 | 61.4 ± 0.7 | 3 |
| episode MIL (gated attention over transients + window embeddings) | 990,676 | 68.4 | 66.4 | 61.7 | 78.8 | 73.5 | 79.2 | 78.9 | 1 |

Best single run on `full`: LLM physical + hand-crafted features + gradient boosting (seed 0), per-class recall: SMALL 84.4, MID 58.1, LARGE 71.9.

### Powertrain: BEV vs ICE

| model | params | window full | window balanced | window human-only | drive full | drive human-only | vehicle full | best val | seeds |
|---|---|---|---|---|---|---|---|---|---|
| hand-crafted features + logistic regression | 0 | 70.7 | 70.8 | 79.4 | 85.3 | 85.6 | - | 67.5 | 1 |
| hand-crafted features + gradient boosting | 0 | 82.7 | 84.8 | 88.9 | 90.7 | 93.1 | - | 82.1 | 1 |
| LLM physical features + logistic regression | 0 | 64.1 | 62.7 | 83.9 | 91.7 | 90.4 | - | 64.8 | 1 |
| LLM physical features + gradient boosting | 0 | 74.8 | 76.5 | 90.4 | 95.4 | 94.2 | - | 71.5 | 1 |
| LLM physical + hand-crafted features + gradient boosting | 0 | 84.4 | 86.3 | 92.4 | 94.1 | 95.7 | - | 84.0 | 1 |
| 1-D ResNet (10 Hz) | 2,308,994 | 82.4 ± 0.7 | 83.2 ± 0.4 | 93.2 ± 0.5 | 96.2 ± 0.5 | 96.9 ± 0.5 | - | 79.9 ± 0.1 | 3 |
| bi-GRU | 727,810 | 87.4 ± 0.4 | 88.3 ± 0.2 | 93.7 ± 0.6 | 96.5 ± 0.1 | 97.9 ± 0.4 | - | 85.3 ± 0.2 | 3 |
| Transformer encoder | 696,962 | 86.1 ± 0.4 | 87.0 ± 0.2 | 93.8 ± 0.2 | 95.2 ± 0.7 | 97.3 ± 0.1 | - | 85.8 ± 0.4 | 3 |
| Mamba-2 (SSD), bidirectional | 1,073,762 | 87.0 | 87.3 | 92.7 | 96.7 | 97.4 | - | 85.8 | 1 |
| bi-GRU, recipe B5 (model-balanced sampling, aux heads with hybrids, augmentation, DAS down-weighting) | 729,352 | 78.8 ± 0.4 | 79.3 ± 0.5 | 92.1 ± 0.3 | 95.1 ± 0.3 | 96.2 ± 0.3 | 96.3 ± 0.0 | 74.8 ± 0.3 | 3 |
| 1-D ResNet, recipe B5 | 2,310,536 | 83.1 ± 0.2 | 83.7 ± 0.3 | 93.9 ± 0.3 | 97.2 ± 0.3 | 97.4 ± 0.4 | 99.0 ± 0.4 | 78.4 ± 0.4 | 3 |
| episode MIL (gated attention over transients + window embeddings) | 990,676 | 86.5 | 87.6 | 93.2 | 96.1 | 97.9 | 97.0 | 90.8 | 1 |

Best single run on `full`: bi-GRU (seed 0), per-class recall: BEV 84.8, ICE 90.6.

## Set-up

- inputs: the 20 official channels (vtt.config.OFFICIAL_INPUTS) of each 600-sample window, clipped per channel, standardised with training-split statistics, NaN -> 0 (lead channels carry `lead_valid`);
- sequence models share a 2-layer convolutional stem (stride 4 -> 150 tokens of width 128); 4 layers for the Transformer and the Mamba variants, 2 bidirectional GRU layers; the ResNet works on the 10 Hz signal;
- training: class-balanced sampling ((N / K n_k)^0.5, the dataset's train_uniform rule), AdamW, warm-up + cosine, label smoothing 0.05, batch 256, at most 25 epochs, early stopping (patience 6) on the validation macro-F1 of the same split scheme;
- the Mamba blocks are pure-PyTorch implementations of the published algorithms (chunked state-space duality scan for Mamba-2, selective scan for Mamba-1), bidirectional (forward and reversed scans summed) as usual for classification.
