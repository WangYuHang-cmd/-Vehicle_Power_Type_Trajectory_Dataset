# In-distribution reference of the seed ensembles (split_platform test, held-out car models)

Generated 9 Nov 2026 21:29 by experiments/transfer/internal_reference.py from results/<tag>_probs.npz. Ensemble = mean of the seeds' window probabilities (the same rule score.py applies to external sets); drive = mean probability per drive, vehicle = all drives of a vehicle pooled; accuracy and macro-F1 in percent, recall per class in the class order. `all` = test_full; `das` = windows with the driving automation engaged on >= 50 % of the samples (the reference for ACC-driven external sets); `human` = strict human-only windows (test_human_only). Rows `full ... + mask-Vk` = the full 20-channel model run on the same windows with the external channel mask Vk applied (missing channels at the training missing value), the reference for the secondary external numbers.

## size (SMALL / MID / LARGE)

| model | subset | level | units | accuracy % | macro-F1 % | majority % | recall per class % | support |
|---|---|---|---|---|---|---|---|---|
| full (GRU S9, 20 ch) | all | window | 26144 | 60.0 | 54.3 | 55.0 | SMALL 71.7 / MID 48.2 / LARGE 42.5 | 14374/6395/5375 |
| full (GRU S9, 20 ch) | all | drive | 1141 | 70.8 | 67.3 | 52.4 | SMALL 80.6 / MID 61.3 / LARGE 58.6 | 598/292/251 |
| full (GRU S9, 20 ch) | all | vehicle | 329 | 63.5 | 59.7 | 48.3 | SMALL 79.2 / MID 47.3 / LARGE 50.6 | 159/93/77 |
| full (GRU S9, 20 ch) | das | window | 14321 | 62.1 | 53.2 | 59.5 | SMALL 74.9 / MID 52.0 / LARGE 33.1 | 8518/3152/2651 |
| full (GRU S9, 20 ch) | das | drive | 816 | 70.2 | 62.5 | 56.0 | SMALL 85.6 / MID 59.7 / LARGE 38.6 | 457/206/153 |
| full (GRU S9, 20 ch) | das | vehicle | 262 | 63.4 | 55.6 | 50.8 | SMALL 86.5 / MID 43.9 / LARGE 34.9 | 133/66/63 |
| full (GRU S9, 20 ch) | human | window | 6138 | 48.9 | 48.2 | 41.6 | SMALL 57.4 / MID 36.8 / LARGE 50.8 | 2556/2025/1557 |
| full (GRU S9, 20 ch) | human | drive | 732 | 63.5 | 62.6 | 46.2 | SMALL 66.9 / MID 61.0 / LARGE 60.3 | 338/195/199 |
| full (GRU S9, 20 ch) | human | vehicle | 269 | 59.5 | 58.6 | 46.1 | SMALL 62.1 / MID 58.5 / LARGE 55.6 | 124/82/63 |
| V1 lateral + speed (10 ch) | all | window | 26144 | 62.0 | 55.0 | 55.0 | SMALL 78.7 / MID 38.3 / LARGE 45.5 | 14374/6395/5375 |
| V1 lateral + speed (10 ch) | all | drive | 1141 | 70.6 | 66.4 | 52.4 | SMALL 83.9 / MID 50.0 / LARGE 62.5 | 598/292/251 |
| V1 lateral + speed (10 ch) | all | vehicle | 329 | 62.6 | 58.0 | 48.3 | SMALL 80.5 / MID 43.0 / LARGE 49.4 | 159/93/77 |
| V1 lateral + speed (10 ch) | das | window | 14321 | 61.2 | 49.8 | 59.5 | SMALL 80.6 / MID 34.0 / LARGE 31.2 | 8518/3152/2651 |
| V1 lateral + speed (10 ch) | das | drive | 816 | 65.1 | 57.4 | 56.0 | SMALL 83.2 / MID 43.2 / LARGE 40.5 | 457/206/153 |
| V1 lateral + speed (10 ch) | das | vehicle | 262 | 62.6 | 54.9 | 50.8 | SMALL 85.0 / MID 45.5 / LARGE 33.3 | 133/66/63 |
| V1 lateral + speed (10 ch) | human | window | 6138 | 55.4 | 54.1 | 41.6 | SMALL 66.1 / MID 37.5 / LARGE 61.1 | 2556/2025/1557 |
| V1 lateral + speed (10 ch) | human | drive | 732 | 67.6 | 65.9 | 46.2 | SMALL 74.9 / MID 56.4 / LARGE 66.3 | 338/195/199 |
| V1 lateral + speed (10 ch) | human | vehicle | 269 | 64.3 | 62.4 | 46.1 | SMALL 73.4 / MID 53.7 / LARGE 60.3 | 124/82/63 |
| V2 longitudinal-only (11 ch) | all | window | 26144 | 41.7 | 34.9 | 55.0 | SMALL 55.8 / MID 18.4 / LARGE 31.6 | 14374/6395/5375 |
| V2 longitudinal-only (11 ch) | all | drive | 1141 | 45.2 | 33.5 | 52.4 | SMALL 72.1 / MID 9.2 / LARGE 23.1 | 598/292/251 |
| V2 longitudinal-only (11 ch) | all | vehicle | 329 | 43.2 | 32.8 | 48.3 | SMALL 73.6 / MID 10.8 / LARGE 19.5 | 159/93/77 |
| V2 longitudinal-only (11 ch) | das | window | 14321 | 43.0 | 35.0 | 59.5 | SMALL 55.3 / MID 22.1 / LARGE 28.2 | 8518/3152/2651 |
| V2 longitudinal-only (11 ch) | das | drive | 816 | 46.8 | 34.3 | 56.0 | SMALL 70.5 / MID 14.1 / LARGE 20.3 | 457/206/153 |
| V2 longitudinal-only (11 ch) | das | vehicle | 262 | 46.2 | 36.0 | 50.8 | SMALL 72.9 / MID 16.7 / LARGE 20.6 | 133/66/63 |
| V2 longitudinal-only (11 ch) | human | window | 6138 | 36.7 | 33.2 | 41.6 | SMALL 55.3 / MID 11.5 / LARGE 38.9 | 2556/2025/1557 |
| V2 longitudinal-only (11 ch) | human | drive | 732 | 41.3 | 36.4 | 46.2 | SMALL 59.8 / MID 14.9 / LARGE 35.7 | 338/195/199 |
| V2 longitudinal-only (11 ch) | human | vehicle | 269 | 39.0 | 34.5 | 46.1 | SMALL 58.9 / MID 13.4 / LARGE 33.3 | 124/82/63 |
| V3 speed-only (3 ch) | all | window | 26144 | 47.1 | 41.3 | 55.0 | SMALL 58.5 / MID 36.7 / LARGE 29.1 | 14374/6395/5375 |
| V3 speed-only (3 ch) | all | drive | 1141 | 53.4 | 45.1 | 52.4 | SMALL 73.4 / MID 37.7 / LARGE 23.9 | 598/292/251 |
| V3 speed-only (3 ch) | all | vehicle | 329 | 54.4 | 45.6 | 48.3 | SMALL 82.4 / MID 32.3 / LARGE 23.4 | 159/93/77 |
| V3 speed-only (3 ch) | das | window | 14321 | 47.3 | 40.1 | 59.5 | SMALL 57.8 / MID 33.7 / LARGE 29.8 | 8518/3152/2651 |
| V3 speed-only (3 ch) | das | drive | 816 | 49.4 | 39.3 | 56.0 | SMALL 69.1 / MID 25.7 / LARGE 22.2 | 457/206/153 |
| V3 speed-only (3 ch) | das | vehicle | 262 | 56.1 | 47.0 | 50.8 | SMALL 82.0 / MID 30.3 / LARGE 28.6 | 133/66/63 |
| V3 speed-only (3 ch) | human | window | 6138 | 43.9 | 41.7 | 41.6 | SMALL 57.7 / MID 34.2 / LARGE 33.8 | 2556/2025/1557 |
| V3 speed-only (3 ch) | human | drive | 732 | 50.4 | 46.5 | 46.2 | SMALL 66.3 / MID 43.6 / LARGE 30.2 | 338/195/199 |
| V3 speed-only (3 ch) | human | vehicle | 269 | 48.3 | 44.1 | 46.1 | SMALL 63.7 / MID 41.5 / LARGE 27.0 | 124/82/63 |
| full (GRU S9, 20 ch) + mask-V1 | all | window | 26144 | 63.2 | 51.2 | 55.0 | SMALL 90.6 / MID 28.3 / LARGE 31.4 | 14374/6395/5375 |
| full (GRU S9, 20 ch) + mask-V1 | all | drive | 1141 | 70.5 | 63.2 | 52.4 | SMALL 95.2 / MID 40.4 / LARGE 46.6 | 598/292/251 |
| full (GRU S9, 20 ch) + mask-V1 | all | vehicle | 329 | 60.8 | 50.8 | 48.3 | SMALL 95.0 / MID 29.0 / LARGE 28.6 | 159/93/77 |
| full (GRU S9, 20 ch) + mask-V1 | das | window | 14321 | 63.4 | 42.3 | 59.5 | SMALL 93.4 / MID 27.9 / LARGE 9.1 | 8518/3152/2651 |
| full (GRU S9, 20 ch) + mask-V1 | das | drive | 816 | 62.7 | 43.4 | 56.0 | SMALL 95.8 / MID 29.1 / LARGE 9.2 | 457/206/153 |
| full (GRU S9, 20 ch) + mask-V1 | das | vehicle | 262 | 55.3 | 36.6 | 50.8 | SMALL 95.5 / MID 22.7 / LARGE 4.8 | 133/66/63 |
| full (GRU S9, 20 ch) + mask-V1 | human | window | 6138 | 55.8 | 51.4 | 41.6 | SMALL 81.7 / MID 21.2 / LARGE 58.2 | 2556/2025/1557 |
| full (GRU S9, 20 ch) + mask-V1 | human | drive | 732 | 68.4 | 65.0 | 46.2 | SMALL 87.9 / MID 43.1 / LARGE 60.3 | 338/195/199 |
| full (GRU S9, 20 ch) + mask-V1 | human | vehicle | 269 | 64.7 | 60.0 | 46.1 | SMALL 91.9 / MID 31.7 / LARGE 54.0 | 124/82/63 |
| full (GRU S9, 20 ch) + mask-V2 | all | window | 26144 | 53.6 | 33.2 | 55.0 | SMALL 89.4 / MID 10.2 / LARGE 9.3 | 14374/6395/5375 |
| full (GRU S9, 20 ch) + mask-V2 | all | drive | 1141 | 52.7 | 24.8 | 52.4 | SMALL 99.2 / MID 1.4 / LARGE 1.6 | 598/292/251 |
| full (GRU S9, 20 ch) + mask-V2 | all | vehicle | 329 | 48.0 | 23.2 | 48.3 | SMALL 98.1 / MID 0.0 / LARGE 2.6 | 159/93/77 |
| full (GRU S9, 20 ch) + mask-V2 | das | window | 14321 | 57.2 | 37.9 | 59.5 | SMALL 85.7 / MID 18.4 / LARGE 11.8 | 8518/3152/2651 |
| full (GRU S9, 20 ch) + mask-V2 | das | drive | 816 | 56.6 | 28.3 | 56.0 | SMALL 98.2 / MID 4.4 / LARGE 2.6 | 457/206/153 |
| full (GRU S9, 20 ch) + mask-V2 | das | vehicle | 262 | 51.1 | 26.3 | 50.8 | SMALL 97.7 / MID 3.0 / LARGE 3.2 | 133/66/63 |
| full (GRU S9, 20 ch) + mask-V2 | human | window | 6138 | 42.1 | 25.5 | 41.6 | SMALL 94.2 / MID 1.8 / LARGE 9.0 | 2556/2025/1557 |
| full (GRU S9, 20 ch) + mask-V2 | human | drive | 732 | 46.0 | 22.2 | 46.2 | SMALL 98.5 / MID 0.0 / LARGE 2.0 | 338/195/199 |
| full (GRU S9, 20 ch) + mask-V2 | human | vehicle | 269 | 46.5 | 23.0 | 46.1 | SMALL 99.2 / MID 0.0 / LARGE 3.2 | 124/82/63 |
| full (GRU S9, 20 ch) + mask-V3 | all | window | 26144 | 55.1 | 30.1 | 55.0 | SMALL 95.3 / MID 8.7 / LARGE 2.9 | 14374/6395/5375 |
| full (GRU S9, 20 ch) + mask-V3 | all | drive | 1141 | 52.8 | 26.1 | 52.4 | SMALL 98.2 / MID 4.8 / LARGE 0.4 | 598/292/251 |
| full (GRU S9, 20 ch) + mask-V3 | all | vehicle | 329 | 49.2 | 25.3 | 48.3 | SMALL 98.7 / MID 4.3 / LARGE 1.3 | 159/93/77 |
| full (GRU S9, 20 ch) + mask-V3 | das | window | 14321 | 59.4 | 30.7 | 59.5 | SMALL 96.3 / MID 6.3 / LARGE 3.7 | 8518/3152/2651 |
| full (GRU S9, 20 ch) + mask-V3 | das | drive | 816 | 56.0 | 25.5 | 56.0 | SMALL 98.9 / MID 1.9 / LARGE 0.7 | 457/206/153 |
| full (GRU S9, 20 ch) + mask-V3 | das | vehicle | 262 | 51.5 | 24.6 | 50.8 | SMALL 100.0 / MID 1.5 / LARGE 1.6 | 133/66/63 |
| full (GRU S9, 20 ch) + mask-V3 | human | window | 6138 | 42.3 | 25.0 | 41.6 | SMALL 94.2 / MID 8.3 / LARGE 1.3 | 2556/2025/1557 |
| full (GRU S9, 20 ch) + mask-V3 | human | drive | 732 | 47.1 | 26.1 | 46.2 | SMALL 97.0 / MID 8.7 / LARGE 0.0 | 338/195/199 |
| full (GRU S9, 20 ch) + mask-V3 | human | vehicle | 269 | 46.8 | 24.7 | 46.1 | SMALL 97.6 / MID 6.1 / LARGE 0.0 | 124/82/63 |

## bev_ice (BEV / ICE)

| model | subset | level | units | accuracy % | macro-F1 % | majority % | recall per class % | support |
|---|---|---|---|---|---|---|---|---|
| full (ResNet B5, 20 ch) | all | window | 18295 | 82.5 | 80.7 | 66.1 | BEV 76.2 / ICE 85.7 | 6194/12101 |
| full (ResNet B5, 20 ch) | all | drive | 836 | 93.7 | 93.3 | 60.9 | BEV 88.7 / ICE 96.9 | 327/509 |
| full (ResNet B5, 20 ch) | all | vehicle | 244 | 93.9 | 93.4 | 61.9 | BEV 89.2 / ICE 96.7 | 93/151 |
| full (ResNet B5, 20 ch) | das | window | 9572 | 76.8 | 75.0 | 64.1 | BEV 69.9 / ICE 80.7 | 3439/6133 |
| full (ResNet B5, 20 ch) | das | drive | 582 | 92.3 | 91.9 | 58.8 | BEV 87.5 / ICE 95.6 | 240/342 |
| full (ResNet B5, 20 ch) | das | vehicle | 192 | 93.2 | 92.8 | 61.5 | BEV 89.2 / ICE 95.8 | 74/118 |
| full (ResNet B5, 20 ch) | human | window | 4669 | 87.3 | 84.0 | 74.1 | BEV 80.7 / ICE 89.6 | 1210/3459 |
| full (ResNet B5, 20 ch) | human | drive | 564 | 90.2 | 89.3 | 65.8 | BEV 88.1 / ICE 91.4 | 193/371 |
| full (ResNet B5, 20 ch) | human | vehicle | 206 | 90.8 | 90.2 | 61.7 | BEV 87.3 / ICE 92.9 | 79/127 |
| V1 lateral + speed (10 ch) | all | window | 18295 | 70.0 | 67.9 | 66.1 | BEV 66.2 / ICE 71.9 | 6194/12101 |
| V1 lateral + speed (10 ch) | all | drive | 836 | 74.6 | 73.7 | 60.9 | BEV 70.6 / ICE 77.2 | 327/509 |
| V1 lateral + speed (10 ch) | all | vehicle | 244 | 75.8 | 74.6 | 61.9 | BEV 71.0 / ICE 78.8 | 93/151 |
| V1 lateral + speed (10 ch) | das | window | 9572 | 69.2 | 68.0 | 64.1 | BEV 70.0 / ICE 68.7 | 3439/6133 |
| V1 lateral + speed (10 ch) | das | drive | 582 | 73.9 | 73.6 | 58.8 | BEV 77.9 / ICE 71.1 | 240/342 |
| V1 lateral + speed (10 ch) | das | vehicle | 192 | 77.6 | 76.8 | 61.5 | BEV 77.0 / ICE 78.0 | 74/118 |
| V1 lateral + speed (10 ch) | human | window | 4669 | 72.6 | 67.2 | 74.1 | BEV 61.9 / ICE 76.4 | 1210/3459 |
| V1 lateral + speed (10 ch) | human | drive | 564 | 73.8 | 71.5 | 65.8 | BEV 66.8 / ICE 77.4 | 193/371 |
| V1 lateral + speed (10 ch) | human | vehicle | 206 | 72.3 | 70.7 | 61.7 | BEV 63.3 / ICE 78.0 | 79/127 |
| V2 longitudinal-only (11 ch) | all | window | 18295 | 76.7 | 74.1 | 66.1 | BEV 66.2 / ICE 82.0 | 6194/12101 |
| V2 longitudinal-only (11 ch) | all | drive | 836 | 90.0 | 89.0 | 60.9 | BEV 78.3 / ICE 97.4 | 327/509 |
| V2 longitudinal-only (11 ch) | all | vehicle | 244 | 91.4 | 90.4 | 61.9 | BEV 77.4 / ICE 100.0 | 93/151 |
| V2 longitudinal-only (11 ch) | das | window | 9572 | 75.3 | 73.0 | 64.1 | BEV 64.3 / ICE 81.4 | 3439/6133 |
| V2 longitudinal-only (11 ch) | das | drive | 582 | 86.9 | 86.3 | 58.8 | BEV 78.3 / ICE 93.0 | 240/342 |
| V2 longitudinal-only (11 ch) | das | vehicle | 192 | 85.4 | 84.2 | 61.5 | BEV 74.3 / ICE 92.4 | 74/118 |
| V2 longitudinal-only (11 ch) | human | window | 4669 | 77.3 | 72.9 | 74.1 | BEV 71.4 / ICE 79.4 | 1210/3459 |
| V2 longitudinal-only (11 ch) | human | drive | 564 | 86.0 | 84.2 | 65.8 | BEV 76.7 / ICE 90.8 | 193/371 |
| V2 longitudinal-only (11 ch) | human | vehicle | 206 | 89.8 | 88.7 | 61.7 | BEV 75.9 / ICE 98.4 | 79/127 |
| V3 speed-only (3 ch) | all | window | 18295 | 77.0 | 74.6 | 66.1 | BEV 68.2 / ICE 81.5 | 6194/12101 |
| V3 speed-only (3 ch) | all | drive | 836 | 89.5 | 88.6 | 60.9 | BEV 78.9 / ICE 96.3 | 327/509 |
| V3 speed-only (3 ch) | all | vehicle | 244 | 91.0 | 90.1 | 61.9 | BEV 79.6 / ICE 98.0 | 93/151 |
| V3 speed-only (3 ch) | das | window | 9572 | 75.1 | 73.4 | 64.1 | BEV 69.0 / ICE 78.5 | 3439/6133 |
| V3 speed-only (3 ch) | das | drive | 582 | 88.3 | 87.9 | 58.8 | BEV 84.6 / ICE 90.9 | 240/342 |
| V3 speed-only (3 ch) | das | vehicle | 192 | 86.5 | 85.6 | 61.5 | BEV 81.1 / ICE 89.8 | 74/118 |
| V3 speed-only (3 ch) | human | window | 4669 | 78.0 | 73.0 | 74.1 | BEV 67.7 / ICE 81.6 | 1210/3459 |
| V3 speed-only (3 ch) | human | drive | 564 | 87.6 | 85.9 | 65.8 | BEV 77.2 / ICE 93.0 | 193/371 |
| V3 speed-only (3 ch) | human | vehicle | 206 | 89.8 | 88.6 | 61.7 | BEV 74.7 / ICE 99.2 | 79/127 |
| full (ResNet B5, 20 ch) + mask-V1 | all | window | 18295 | 68.9 | 62.4 | 66.1 | BEV 40.3 / ICE 83.6 | 6194/12101 |
| full (ResNet B5, 20 ch) + mask-V1 | all | drive | 836 | 71.2 | 66.7 | 60.9 | BEV 44.3 / ICE 88.4 | 327/509 |
| full (ResNet B5, 20 ch) + mask-V1 | all | vehicle | 244 | 69.3 | 61.2 | 61.9 | BEV 31.2 / ICE 92.7 | 93/151 |
| full (ResNet B5, 20 ch) + mask-V1 | das | window | 9572 | 73.5 | 67.1 | 64.1 | BEV 41.0 / ICE 91.7 | 3439/6133 |
| full (ResNet B5, 20 ch) + mask-V1 | das | drive | 582 | 75.1 | 71.0 | 58.8 | BEV 45.4 / ICE 95.9 | 240/342 |
| full (ResNet B5, 20 ch) + mask-V1 | das | vehicle | 192 | 76.6 | 71.8 | 61.5 | BEV 45.9 / ICE 95.8 | 74/118 |
| full (ResNet B5, 20 ch) + mask-V1 | human | window | 4669 | 64.7 | 55.4 | 74.1 | BEV 36.8 / ICE 74.5 | 1210/3459 |
| full (ResNet B5, 20 ch) + mask-V1 | human | drive | 564 | 62.8 | 56.1 | 65.8 | BEV 34.7 / ICE 77.4 | 193/371 |
| full (ResNet B5, 20 ch) + mask-V1 | human | vehicle | 206 | 59.2 | 52.8 | 61.7 | BEV 29.1 / ICE 78.0 | 79/127 |
| full (ResNet B5, 20 ch) + mask-V2 | all | window | 18295 | 71.0 | 60.9 | 66.1 | BEV 29.8 / ICE 92.0 | 6194/12101 |
| full (ResNet B5, 20 ch) + mask-V2 | all | drive | 836 | 67.5 | 54.4 | 60.9 | BEV 17.7 / ICE 99.4 | 327/509 |
| full (ResNet B5, 20 ch) + mask-V2 | all | vehicle | 244 | 69.3 | 56.8 | 61.9 | BEV 20.4 / ICE 99.3 | 93/151 |
| full (ResNet B5, 20 ch) + mask-V2 | das | window | 9572 | 71.4 | 60.9 | 64.1 | BEV 27.2 / ICE 96.2 | 3439/6133 |
| full (ResNet B5, 20 ch) + mask-V2 | das | drive | 582 | 66.0 | 53.7 | 58.8 | BEV 17.5 / ICE 100.0 | 240/342 |
| full (ResNet B5, 20 ch) + mask-V2 | das | vehicle | 192 | 66.7 | 51.2 | 61.5 | BEV 13.5 / ICE 100.0 | 74/118 |
| full (ResNet B5, 20 ch) + mask-V2 | human | window | 4669 | 72.9 | 61.6 | 74.1 | BEV 35.9 / ICE 85.9 | 1210/3459 |
| full (ResNet B5, 20 ch) + mask-V2 | human | drive | 564 | 72.2 | 59.1 | 65.8 | BEV 22.8 / ICE 97.8 | 193/371 |
| full (ResNet B5, 20 ch) + mask-V2 | human | vehicle | 206 | 69.9 | 58.5 | 61.7 | BEV 22.8 / ICE 99.2 | 79/127 |
| full (ResNet B5, 20 ch) + mask-V3 | all | window | 18295 | 69.7 | 59.4 | 66.1 | BEV 28.6 / ICE 90.8 | 6194/12101 |
| full (ResNet B5, 20 ch) + mask-V3 | all | drive | 836 | 63.3 | 51.0 | 60.9 | BEV 16.8 / ICE 93.1 | 327/509 |
| full (ResNet B5, 20 ch) + mask-V3 | all | vehicle | 244 | 65.2 | 49.8 | 61.9 | BEV 12.9 / ICE 97.4 | 93/151 |
| full (ResNet B5, 20 ch) + mask-V3 | das | window | 9572 | 70.3 | 62.0 | 64.1 | BEV 32.7 / ICE 91.5 | 3439/6133 |
| full (ResNet B5, 20 ch) + mask-V3 | das | drive | 582 | 69.4 | 60.7 | 58.8 | BEV 27.1 / ICE 99.1 | 240/342 |
| full (ResNet B5, 20 ch) + mask-V3 | das | vehicle | 192 | 70.3 | 59.0 | 61.5 | BEV 23.0 / ICE 100.0 | 74/118 |
| full (ResNet B5, 20 ch) + mask-V3 | human | window | 4669 | 69.8 | 54.5 | 74.1 | BEV 22.7 / ICE 86.3 | 1210/3459 |
| full (ResNet B5, 20 ch) + mask-V3 | human | drive | 564 | 64.0 | 49.4 | 65.8 | BEV 15.0 / ICE 89.5 | 193/371 |
| full (ResNet B5, 20 ch) + mask-V3 | human | vehicle | 206 | 63.6 | 50.1 | 61.7 | BEV 15.2 / ICE 93.7 | 79/127 |

