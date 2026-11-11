# Zero-shot transfer: generated tables

Generated 10 Nov 2026 21:06 by experiments/transfer/collect_scores.py from the score__*.json / probs__*__ens__*.npz files of score.py and internal_reference.json. Accuracy and macro-F1 in percent (one decimal); 95 % intervals from the vehicle-cluster bootstrap (1,000 resamples; absent when a set has fewer than about 20 bootstrap-able clusters or one vehicle). Chance = 1 / K and the majority share.

## Cross-dataset summary

### bev_ice (BEV / ICE): primary run per dataset

| dataset | model | labelled windows / drives / vehicles / models | window acc / F1 % | drive acc [95% CI] / F1 % | vehicle acc [95% CI] / F1 % | vehicle-level recall per class % | chance uniform / majority % | internal reference drive acc / F1 / vehicle acc / F1 (same variant) |
|---|---|---|---|---|---|---|---|---|
| ultraav | V2 variant | 1987 / 84 / 12 / 20 | 72.1 / 63.3 | 85.7 [46.1, 96.7] / 46.2 | 75.0 [50.0, 100.0] / 69.7 | BEV 100.0 / ICE 40.0 | 50.0 / 66.9 | 90.0 / 89.0 / 91.4 / 90.4 (das: 86.9 / 86.3 / 85.4 / 84.2) |
| openacc | V2 variant | 3955 / 512 / 19 / 20 | 68.2 / 67.6 | 80.3 [64.2, 90.6] / 76.3 | 73.7 [52.6, 94.7] / 73.7 | BEV 100.0 / ICE 58.3 | 50.0 / 52.5 | 90.0 / 89.0 / 91.4 / 90.4 (das: 86.9 / 86.3 / 85.4 / 84.2) |
| microsim_ev | V2 variant | 111 / 110 / 3 / 3 | 55.0 / - | 54.5 [50.0, 100.0] / - | 100.0 [100.0, 100.0] / - | BEV 100.0 / ICE - | 50.0 / 100.0 | 90.0 / 89.0 / 91.4 / 90.4 (das: 86.9 / 86.3 / 85.4 / 84.2) |
| uga_ev | V2 variant | 326 / 48 / 4 / 3 | 42.6 / 37.6 | 47.9 [25.0, 70.8] / 42.3 | 50.0 [0.0, 100.0] / 33.3 | BEV 66.7 / ICE 0.0 | 50.0 / 74.8 | 90.0 / 89.0 / 91.4 / 90.4 (das: 86.9 / 86.3 / 85.4 / 84.2) |
| tesla_yuan | V3 variant | 2619 / 106 / 1 / 1 | 62.3 / - | 80.2 [80.2, 80.2] / - | 100.0 [100.0, 100.0] / - | BEV 100.0 / ICE - | 50.0 / 100.0 | 89.5 / 88.6 / 91.0 / 90.1 (human: 87.6 / 85.9 / 89.8 / 88.6) |
| openacc_500x | V3 variant | 13523 / 429 / 1 / 1 | 87.9 / - | 99.1 [99.1, 99.1] / - | 100.0 [100.0, 100.0] / - | BEV - / ICE 100.0 | 50.0 / 100.0 | 89.5 / 88.6 / 91.0 / 90.1 (human: 87.6 / 85.9 / 89.8 / 88.6) |
| ved | V3 variant | 39259 / 15770 / 267 / 21 | 68.8 / 41.3 | 74.6 [72.3, 76.6] / 44.0 | 96.6 [94.4, 98.5] / 49.1 | BEV 0.0 / ICE 97.7 | 50.0 / 98.9 | 89.5 / 88.6 / 91.0 / 90.1 (human: 87.6 / 85.9 / 89.8 / 88.6) |
| a2d2 | full 20-ch model | 0 / - / - / 1 | - / - | - / - | - / - | - | 50.0 / - | 93.7 / 93.3 / 93.9 / 93.4 (human: 90.2 / 89.3 / 90.8 / 90.2) |
| ccs_sample | full 20-ch model | 4 / 4 / 4 / 4 | 75.0 / 42.9 | 75.0 [25.0, 100.0] / 42.9 | 75.0 [25.0, 100.0] / 42.9 | BEV 100.0 / ICE 0.0 | 50.0 / 75.0 | 93.7 / 93.3 / 93.9 / 93.4 |
| csc | V3 variant | 51462 / 51462 / 36 / 25 | 75.4 / 69.0 | 75.4 [72.2, 78.5] / 69.0 | 97.2 [91.7, 100.0] / 96.1 | BEV 88.9 / ICE 100.0 | 50.0 / 76.8 | 89.5 / 88.6 / 91.0 / 90.1 (das: 88.3 / 87.9 / 86.5 / 85.6) |
| moretorque | full 20-ch model | 7190 / 1155 / 12 / 4 | 72.4 / 45.2 | 85.9 [66.1, 94.4] / 48.5 | 91.7 [75.0, 100.0] / 47.8 | BEV 0.0 / ICE 100.0 | 50.0 / 87.8 | 93.7 / 93.3 / 93.9 / 93.4 |
| ccs | full 20-ch model | 29585 / 14355 / 2512 / 113 | 81.8 / 74.7 | 86.6 [85.7, 87.5] / 79.4 | 87.8 [86.5, 89.0] / 84.7 | BEV 86.6 / ICE 88.2 | 50.0 / 81.7 | 93.7 / 93.3 / 93.9 / 93.4 |
| ccs_honda | full 20-ch model | 2084 / 883 / 229 / 5 | 89.3 / - | 94.1 [91.9, 96.1] / - | 99.1 [97.8, 100.0] / - | BEV - / ICE 99.1 | 50.0 / 100.0 | 93.7 / 93.3 / 93.9 / 93.4 |
| comma2k19 | full 20-ch model | 381 / 42 / 2 / 2 | 70.9 / - | 88.1 [85.7, 90.5] / - | 100.0 [100.0, 100.0] / - | BEV - / ICE 100.0 | 50.0 / 100.0 | 93.7 / 93.3 / 93.9 / 93.4 |

### size (SMALL / MID / LARGE): primary run per dataset

| dataset | model | labelled windows / drives / vehicles / models | window acc / F1 % | drive acc [95% CI] / F1 % | vehicle acc [95% CI] / F1 % | vehicle-level recall per class % | chance uniform / majority % | internal reference drive acc / F1 / vehicle acc / F1 (same variant) |
|---|---|---|---|---|---|---|---|---|
| ultraav | V3 variant | 2976 / 121 / 22 / 20 | 29.7 / 19.1 | 21.5 [8.5, 33.3] / 14.0 | 36.4 [18.2, 54.5] / 17.8 | SMALL 100.0 / MID 0.0 / LARGE 0.0 | 33.3 / 37.4 | 53.4 / 45.1 / 54.4 / 45.6 (das: 49.4 / 39.3 / 56.1 / 47.0) |
| openacc | V3 variant | 4540 / 630 / 21 / 20 | 30.2 / 22.0 | 27.1 [10.0, 50.8] / 19.9 | 38.1 [19.0, 57.1] / 18.4 | SMALL 88.9 / MID 0.0 / LARGE 0.0 | 33.3 / 39.0 | 53.4 / 45.1 / 54.4 / 45.6 (das: 49.4 / 39.3 / 56.1 / 47.0) |
| microsim_ev | V3 variant | 111 / 110 / 3 / 3 | 26.1 / 25.7 | 26.4 [0.0, 100.0] / 26.0 | 33.3 [0.0, 100.0] / 25.0 | SMALL 100.0 / MID 0.0 / LARGE - | 33.3 / 92.8 | 53.4 / 45.1 / 54.4 / 45.6 (das: 49.4 / 39.3 / 56.1 / 47.0) |
| uga_ev | V3 variant | 326 / 48 / 4 / 3 | 73.9 / 56.7 | 75.0 [25.0, 100.0] / 42.9 | 75.0 [25.0, 100.0] / 42.9 | SMALL 100.0 / MID 0.0 / LARGE - | 33.3 / 74.5 | 53.4 / 45.1 / 54.4 / 45.6 (das: 49.4 / 39.3 / 56.1 / 47.0) |
| tesla_yuan | V3 variant | 2619 / 106 / 1 / 1 | 63.7 / - | 78.3 [78.3, 78.3] / - | 100.0 [100.0, 100.0] / - | SMALL - / MID 100.0 / LARGE - | 33.3 / 100.0 | 53.4 / 45.1 / 54.4 / 45.6 (human: 50.4 / 46.5 / 48.3 / 44.1) |
| openacc_500x | V3 variant | 13523 / 429 / 1 / 1 | 57.3 / - | 46.2 [46.2, 46.2] / - | 100.0 [100.0, 100.0] / - | SMALL 100.0 / MID - / LARGE - | 33.3 / 100.0 | 53.4 / 45.1 / 54.4 / 45.6 (human: 50.4 / 46.5 / 48.3 / 44.1) |
| ved | V3 variant | 431 / 317 / 3 / 21 | 60.3 / - | 62.5 [58.7, 65.9] / - | 100.0 [100.0, 100.0] / - | SMALL 100.0 / MID - / LARGE - | 33.3 / 100.0 | 53.4 / 45.1 / 54.4 / 45.6 (human: 50.4 / 46.5 / 48.3 / 44.1) |
| a2d2 | full 20-ch model | 34 / 3 / 1 / 1 | 0.0 / - | 0.0 [0.0, 0.0] / - | 0.0 [0.0, 0.0] / - | SMALL - / MID - / LARGE 0.0 | 33.3 / 100.0 | 70.8 / 67.3 / 63.5 / 59.7 (human: 63.5 / 62.6 / 59.5 / 58.6) |
| ccs_sample | full 20-ch model | 4 / 4 / 4 / 4 | 50.0 / 38.9 | 50.0 [0.0, 100.0] / 38.9 | 50.0 [0.0, 100.0] / 38.9 | SMALL 50.0 / MID 100.0 / LARGE 0.0 | 33.3 / 50.0 | 70.8 / 67.3 / 63.5 / 59.7 |
| csc | V1 variant | 51462 / 51462 / 36 / 25 | 48.4 / 44.1 | 48.4 [41.5, 56.1] / 44.1 | 63.9 [50.0, 80.6] / 62.3 | SMALL 88.2 / MID 30.8 / LARGE 66.7 | 33.3 / 46.6 | 70.6 / 66.4 / 62.6 / 58.0 (das: 65.1 / 57.4 / 62.6 / 54.9) |
| moretorque | full 20-ch model | 7190 / 1155 / 12 / 4 | 43.4 / 41.1 | 42.7 [17.0, 79.6] / 37.6 | 66.7 [41.7, 91.7] / 40.0 | SMALL 100.0 / MID 0.0 / LARGE - | 33.3 / 55.6 | 70.8 / 67.3 / 63.5 / 59.7 |
| ccs | full 20-ch model | 45763 / 21987 / 4061 / 113 | 49.6 / 43.2 | 51.3 [49.3, 53.2] / 44.0 | 54.1 [52.5, 55.7] / 46.4 | SMALL 91.1 / MID 27.6 / LARGE 23.5 | 33.3 / 43.8 | 70.8 / 67.3 / 63.5 / 59.7 |
| ccs_honda | full 20-ch model | 2132 / 913 / 230 / 5 | 83.8 / - | 88.2 [84.8, 91.5] / - | 98.3 [96.5, 99.6] / - | SMALL 98.3 / MID - / LARGE - | 33.3 / 100.0 | 70.8 / 67.3 / 63.5 / 59.7 |
| comma2k19 | full 20-ch model | 381 / 42 / 2 / 2 | 51.7 / - | 54.8 [52.4, 57.1] / - | 50.0 [0.0, 100.0] / - | SMALL 50.0 / MID - / LARGE - | 33.3 / 100.0 | 70.8 / 67.3 / 63.5 / 59.7 |

### bev_ice: every named external car model, primary run of its dataset (mean probability over all windows of the model -> arg-max)

| dataset | car model | label | powertrain | footprint | veh | win | DAS % | model | P(BEV/ICE) | decision | hit | vehicles correct | drive acc % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ultraav | Audi A6 | ICE | ICE | 4.75 | 1 | 173 | 100.0 | V2 variant | 0.70, 0.30 | BEV | MISS | 0/1 | 0.0 |
| ultraav | Audi e-tron | BEV (border) | BEV | 4.84 | 1 | 219 | 95.4 | V2 variant | 0.88, 0.12 | BEV | OK | 1/1 | 100.0 |
| ultraav | BMW X5 | ICE | ICE | 4.99 | 1 | 173 | 100.0 | V2 variant | 0.60, 0.40 | BEV | MISS | 0/1 | 12.5 |
| ultraav | Ford S-Max | ICE | ICE | 4.57 | 1 | 59 | 100.0 | V2 variant | 0.48, 0.52 | ICE | OK | 1/1 | 42.9 |
| ultraav | Jaguar I-Pace | BEV (border) | BEV | 4.92 | 1 | 195 | 95.9 | V2 variant | 0.85, 0.15 | BEV | OK | 1/1 | 97.8 |
| ultraav | Mazda 3 | ICE | ICE | 4.20 | 1 | 79 | 100.0 | V2 variant | 0.69, 0.30 | BEV | MISS | 0/1 | 10.0 |
| ultraav | Mercedes-Benz A-Class | ICE | ICE | 4.25 | 1 | 173 | 100.0 | V2 variant | 0.48, 0.52 | ICE | OK | 1/1 | 50.0 |
| ultraav | Tesla Model 3 | BEV | BEV | 4.57 | 2 | 388 | 96.9 | V2 variant | 0.77, 0.23 | BEV | OK | 2/2 | 94.3 |
| ultraav | Tesla Model S | BEV | BEV | 4.97 | 1 | 255 | 95.7 | V2 variant | 0.79, 0.21 | BEV | OK | 1/1 | 91.9 |
| ultraav | Tesla Model X | BEV | BEV | 5.06 | 1 | 218 | 98.2 | V2 variant | 0.74, 0.26 | BEV | OK | 1/1 | 94.0 |
| ultraav | Volkswagen e-Golf | BEV | BEV | 4.02 | 1 | 55 | 100.0 | V2 variant | 0.53, 0.47 | BEV | OK | 1/1 | 83.3 |
| openacc | Audi A6 | ICE | ICE | 4.75 | 1 | 220 | 79.1 | V2 variant | 0.72, 0.28 | BEV | MISS | 0/1 | 0.0 |
| openacc | Audi A8 | ICE (border) | ICE | 4.91 | 1 | 220 | 100.0 | V2 variant | 0.22, 0.78 | ICE | OK | 1/1 | 100.0 |
| openacc | Audi e-tron | BEV (border) | BEV | 4.84 | 1 | 350 | 90.3 | V2 variant | 0.79, 0.21 | BEV | OK | 1/1 | 88.1 |
| openacc | BMW X5 | ICE | ICE | 4.99 | 1 | 220 | 79.1 | V2 variant | 0.64, 0.36 | BEV | MISS | 0/1 | 10.0 |
| openacc | Fiat 500X | ICE | ICE | 3.97 | 1 | 182 | 0.0 | V2 variant | 0.30, 0.70 | ICE | OK | 1/1 | 83.3 |
| openacc | Ford S-Max | ICE | ICE | 4.57 | 1 | 217 | 83.1 | V2 variant | 0.43, 0.57 | ICE | OK | 1/1 | 72.2 |
| openacc | Jaguar I-Pace | BEV (border) | BEV | 4.92 | 1 | 292 | 88.7 | V2 variant | 0.79, 0.21 | BEV | OK | 1/1 | 96.7 |
| openacc | MINI Cooper | ICE | ICE | 3.74 | 1 | 42 | 0.0 | V2 variant | 0.39, 0.61 | ICE | OK | 1/1 | 100.0 |
| openacc | Mazda 3 | ICE | ICE | 4.29 | 1 | 305 | 88.9 | V2 variant | 0.54, 0.46 | BEV | MISS | 0/1 | 42.9 |
| openacc | Mercedes-Benz A-Class | ICE | ICE | 4.25 | 1 | 220 | 79.1 | V2 variant | 0.53, 0.47 | BEV | MISS | 0/1 | 30.0 |
| openacc | Mitsubishi Space Star | ICE | ICE | 3.48 | 1 | 114 | 0.0 | V2 variant | 0.28, 0.71 | ICE | OK | 1/1 | 100.0 |
| openacc | SsangYong Rexton | ICE | ICE | 4.70 | 1 | 69 | 0.0 | V2 variant | 0.38, 0.62 | ICE | OK | 1/1 | 66.7 |
| openacc | Tesla Model 3 | BEV | BEV | 4.57 | 2 | 486 | 84.4 | V2 variant | 0.79, 0.21 | BEV | OK | 2/2 | 96.6 |
| openacc | Tesla Model S | BEV | BEV | 4.97 | 1 | 334 | 89.8 | V2 variant | 0.79, 0.21 | BEV | OK | 1/1 | 98.6 |
| openacc | Tesla Model X | BEV | BEV | 5.06 | 1 | 345 | 90.4 | V2 variant | 0.73, 0.27 | BEV | OK | 1/1 | 94.2 |
| openacc | Volkswagen Polo | ICE | ICE | 3.61 | 1 | 82 | 0.0 | V2 variant | 0.30, 0.70 | ICE | OK | 1/1 | 100.0 |
| openacc | Volkswagen e-Golf | BEV | BEV | 4.02 | 1 | 73 | 90.1 | V2 variant | 0.51, 0.49 | BEV | OK | 1/1 | 57.1 |
| openacc | Volvo XC40 | ICE | ICE | 4.36 | 1 | 184 | 86.5 | V2 variant | 0.73, 0.27 | BEV | MISS | 0/1 | 16.7 |
| microsim_ev | Hyundai Ioniq 5 | BEV (border) | BEV | 4.83 | 1 | 99 | 100.0 | V2 variant | 0.54, 0.46 | BEV | OK | 1/1 | 51.0 |
| microsim_ev | Polestar 2 | BEV (border) | BEV | 4.38 | 1 | 8 | 100.0 | V2 variant | 0.92, 0.08 | BEV | OK | 1/1 | 100.0 |
| microsim_ev | Tesla Model 3 | BEV | BEV | 4.57 | 1 | 4 | 100.0 | V2 variant | 0.54, 0.46 | BEV | OK | 1/1 | 50.0 |
| uga_ev | Ford Mustang Mach-E | BEV (border) | BEV | 4.84 | 1 | 83 | 67.5 | V2 variant | 0.36, 0.64 | ICE | MISS | 0/1 | 16.7 |
| uga_ev | Kia Niro EV | BEV | BEV | 4.26 | 2 | 161 | 67.1 | V2 variant | 0.55, 0.45 | BEV | OK | 2/2 | 70.8 |
| uga_ev | Lexus UX 200 | ICE | ICE | 4.09 | 1 | 82 | 0.0 | V2 variant | 0.58, 0.42 | BEV | MISS | 0/1 | 33.3 |
| tesla_yuan | Tesla Model 3 | BEV | BEV | 4.57 | 1 | 2619 | 0.0 | V3 variant | 0.61, 0.39 | BEV | OK | 1/1 | 80.2 |
| openacc_500x | Fiat 500X | ICE | ICE | 3.97 | 1 | 13523 | 0.0 | V3 variant | 0.18, 0.82 | ICE | OK | 1/1 | 99.1 |
| ved | Nissan Leaf (2013) | BEV | BEV | 4.15 | 3 | 431 | 0.0 | V3 variant | 0.26, 0.74 | ICE | MISS | 0/3 | 16.4 |
| ved | VED ICE 2500 lb | ICE | ICE | - | 36 | 5534 | 0.0 | V3 variant | 0.34, 0.66 | ICE | OK | 35/36 | 77.0 |
| ved | VED ICE 3000 lb | ICE | ICE | - | 53 | 7457 | 0.0 | V3 variant | 0.36, 0.64 | ICE | OK | 52/53 | 71.6 |
| ved | VED ICE 3500 lb | ICE | ICE | - | 82 | 13279 | 0.0 | V3 variant | 0.35, 0.65 | ICE | OK | 81/82 | 75.1 |
| ved | VED ICE 4000 lb | ICE | ICE | - | 29 | 4256 | 0.0 | V3 variant | 0.32, 0.68 | ICE | OK | 29/29 | 79.9 |
| ved | VED ICE 4500 lb | ICE | ICE | - | 35 | 4611 | 0.0 | V3 variant | 0.33, 0.67 | ICE | OK | 32/35 | 77.3 |
| ved | VED ICE 5000 lb | ICE | ICE | - | 5 | 577 | 0.0 | V3 variant | 0.31, 0.69 | ICE | OK | 5/5 | 77.9 |
| ved | VED ICE 5500 lb | ICE | ICE | - | 4 | 331 | 0.0 | V3 variant | 0.34, 0.66 | ICE | OK | 4/4 | 77.1 |
| ved | VED ICE 6000 lb | ICE | ICE | - | 3 | 198 | 0.0 | V3 variant | 0.32, 0.68 | ICE | OK | 3/3 | 77.8 |
| ved | VED ICE 6500 lb | ICE | ICE | - | 1 | 15 | 0.0 | V3 variant | 0.24, 0.76 | ICE | OK | 1/1 | 100.0 |
| ved | VED ICE weight n/a | ICE | ICE | - | 16 | 2570 | 0.0 | V3 variant | 0.32, 0.68 | ICE | OK | 16/16 | 80.8 |
| ccs_sample | Chevrolet Bolt EUV | BEV | BEV | 3.99 | 1 | 1 | 35.3 | full 20-ch model | 0.99, 0.01 | BEV | OK | 1/1 | 100.0 |
| ccs_sample | Rivian R1T/R1S | BEV | BEV | 5.58 | 1 | 1 | 99.8 | full 20-ch model | 0.87, 0.13 | BEV | OK | 1/1 | 100.0 |
| ccs_sample | Tesla Model 3 | BEV | BEV | 4.57 | 1 | 1 | 0.0 | full 20-ch model | 0.99, 0.01 | BEV | OK | 1/1 | 100.0 |
| ccs_sample | Toyota RAV4 | ICE | ICE | 4.34 | 1 | 1 | 99.8 | full 20-ch model | 0.83, 0.17 | BEV | MISS | 0/1 | 0.0 |
| csc | Chevrolet Bolt EUV | BEV | BEV | 3.99 | 1 | 1500 | 90.7 | V3 variant | 0.66, 0.34 | BEV | OK | 1/1 | 73.1 |
| csc | Chevrolet Silverado 1500 | ICE | ICE | 6.65 | 1 | 1500 | 92.4 | V3 variant | 0.34, 0.66 | ICE | OK | 1/1 | 73.9 |
| csc | Genesis GV60 | BEV | BEV | 4.73 | 1 | 999 | 94.9 | V3 variant | 0.58, 0.42 | BEV | OK | 1/1 | 62.8 |
| csc | Honda Accord | ICE | ICE | 4.54 | 1 | 1500 | 91.9 | V3 variant | 0.24, 0.76 | ICE | OK | 1/1 | 83.1 |
| csc | Honda CR-V | ICE | ICE | 4.23 | 2 | 2500 | 90.8 | V3 variant | 0.26, 0.74 | ICE | OK | 2/2 | 81.6 |
| csc | Honda Civic | ICE | ICE | 4.20 | 3 | 4500 | 91.3 | V3 variant | 0.28, 0.72 | ICE | OK | 3/3 | 77.9 |
| csc | Honda Odyssey | ICE | ICE | 5.14 | 1 | 1500 | 91.6 | V3 variant | 0.37, 0.63 | ICE | OK | 1/1 | 67.9 |
| csc | Honda Pilot | ICE (border) | ICE | 4.91 | 1 | 1500 | 92.9 | V3 variant | 0.22, 0.78 | ICE | OK | 1/1 | 86.2 |
| csc | Hyundai Elantra | ICE | ICE | 4.31 | 1 | 1500 | 90.6 | V3 variant | 0.31, 0.69 | ICE | OK | 1/1 | 74.4 |
| csc | Hyundai Ioniq | BEV | BEV | 4.22 | 1 | 497 | 92.8 | V3 variant | 0.32, 0.68 | ICE | MISS | 0/1 | 26.2 |
| csc | Hyundai Ioniq 5 | BEV (border) | BEV | 4.83 | 1 | 1500 | 92.3 | V3 variant | 0.66, 0.34 | BEV | OK | 1/1 | 72.3 |
| csc | Hyundai Kona | BEV | BEV | 4.02 | 1 | 1500 | 92.6 | V3 variant | 0.55, 0.45 | BEV | OK | 1/1 | 58.0 |
| csc | Hyundai Palisade | ICE | ICE | 4.96 | 1 | 1500 | 93.3 | V3 variant | 0.33, 0.67 | ICE | OK | 1/1 | 73.0 |
| csc | Hyundai Santa Fe | ICE | ICE | 4.53 | 2 | 3000 | 92.9 | V3 variant | 0.30, 0.70 | ICE | OK | 2/2 | 76.9 |
| csc | Hyundai Sonata | ICE | ICE | 4.56 | 2 | 3000 | 90.7 | V3 variant | 0.32, 0.68 | ICE | OK | 2/2 | 71.9 |
| csc | Hyundai Tucson | ICE | ICE | 4.53 | 1 | 999 | 93.2 | V3 variant | 0.25, 0.75 | ICE | OK | 1/1 | 83.6 |
| csc | Kia EV6 | BEV | BEV | 4.74 | 1 | 1500 | 93.1 | V3 variant | 0.65, 0.35 | BEV | OK | 1/1 | 70.4 |
| csc | Kia Niro | BEV | BEV | 4.27 | 1 | 1500 | 90.4 | V3 variant | 0.60, 0.40 | BEV | OK | 1/1 | 67.5 |
| csc | Nissan Leaf | BEV | BEV | 4.16 | 2 | 2967 | 91.0 | V3 variant | 0.55, 0.45 | BEV | OK | 2/2 | 59.6 |
| csc | RAM 1500 | ICE | ICE | 6.73 | 1 | 1500 | 94.1 | V3 variant | 0.33, 0.67 | ICE | OK | 1/1 | 73.7 |
| csc | Toyota Camry | ICE (border) | ICE | 4.47 | 2 | 3000 | 92.2 | V3 variant | 0.19, 0.81 | ICE | OK | 2/2 | 87.0 |
| csc | Toyota Corolla | ICE | ICE | 4.14 | 2 | 3000 | 91.4 | V3 variant | 0.28, 0.72 | ICE | OK | 2/2 | 77.3 |
| csc | Toyota Highlander | ICE | ICE | 4.60 | 2 | 3000 | 93.1 | V3 variant | 0.26, 0.74 | ICE | OK | 2/2 | 81.5 |
| csc | Toyota RAV4 | ICE | ICE | 4.13 | 3 | 4500 | 92.6 | V3 variant | 0.19, 0.81 | ICE | OK | 3/3 | 89.6 |
| csc | Toyota Sienna | ICE | ICE | 5.21 | 1 | 1500 | 93.1 | V3 variant | 0.40, 0.60 | ICE | OK | 1/1 | 63.3 |
| moretorque | Hyundai Sonata | ICE | ICE | 4.62 | 1 | 2926 | 63.4 | full 20-ch model | 0.31, 0.69 | ICE | OK | 1/1 | 90.6 |
| moretorque | Mazda 3 | ICE | ICE | 4.29 | 7 | 2319 | 72.8 | full 20-ch model | 0.24, 0.76 | ICE | OK | 7/7 | 91.7 |
| moretorque | Mazda CX-50 | ICE | ICE | 4.67 | 3 | 1071 | 14.0 | full 20-ch model | 0.11, 0.89 | ICE | OK | 3/3 | 95.5 |
| moretorque | Omoda E5 | BEV | BEV | 4.05 | 1 | 874 | 29.6 | full 20-ch model | 0.20, 0.80 | ICE | MISS | 0/1 | 4.5 |
| ccs | Acura ILX | ICE | ICE | 4.06 | 1 | 4 | 29.8 | full 20-ch model | 0.04, 0.96 | ICE | OK | 1/1 | 100.0 |
| ccs | Acura RDX | ICE | ICE | 4.50 | 21 | 476 | 35.2 | full 20-ch model | 0.17, 0.83 | ICE | OK | 21/21 | 95.2 |
| ccs | Audi Q3 | ICE | ICE | 4.23 | 11 | 380 | 17.8 | full 20-ch model | 0.16, 0.84 | ICE | OK | 11/11 | 93.2 |
| ccs | Buick Regal Sportback | ICE | ICE | 4.52 | 1 | 50 | 48.9 | full 20-ch model | 0.22, 0.78 | ICE | OK | 1/1 | 87.9 |
| ccs | Cadillac Escalade | ICE | ICE | 5.14 | 1 | 1 | 99.8 | full 20-ch model | 0.93, 0.07 | BEV | MISS | 0/1 | 0.0 |
| ccs | Cadillac Escalade ESV | ICE | ICE | 5.76 | 1 | 8 | 55.5 | full 20-ch model | 0.33, 0.67 | ICE | OK | 1/1 | 100.0 |
| ccs | Cadillac XT4 | ICE (border) | ICE | 4.46 | 1 | 1 | 1.2 | full 20-ch model | 0.03, 0.97 | ICE | OK | 1/1 | 100.0 |
| ccs | Chevrolet Bolt EUV | BEV | BEV | 3.99 | 160 | 380 | 43.3 | full 20-ch model | 0.73, 0.27 | BEV | OK | 118/160 | 73.5 |
| ccs | Chevrolet Equinox | ICE | ICE | 4.30 | 5 | 243 | 45.7 | full 20-ch model | 0.20, 0.80 | ICE | OK | 5/5 | 89.1 |
| ccs | Chevrolet Silverado 1500 | ICE | ICE | 6.65 | 52 | 395 | 48.3 | full 20-ch model | 0.21, 0.79 | ICE | OK | 50/52 | 94.9 |
| ccs | Chevrolet Traverse | ICE | ICE | 5.24 | 1 | 29 | 47.5 | full 20-ch model | 0.16, 0.84 | ICE | OK | 1/1 | 100.0 |
| ccs | Chrysler Pacifica | ICE | ICE | 5.36 | 59 | 709 | 28.4 | full 20-ch model | 0.17, 0.83 | ICE | OK | 56/59 | 91.6 |
| ccs | Dodge Durango | ICE | ICE | 4.95 | 1 | 11 | 0.0 | full 20-ch model | 0.16, 0.84 | ICE | OK | 1/1 | 100.0 |
| ccs | Ford Bronco Sport | ICE | ICE | 4.28 | 8 | 94 | 60.5 | full 20-ch model | 0.38, 0.62 | ICE | OK | 7/8 | 80.0 |
| ccs | Ford F-150 Lightning | BEV | BEV | 6.41 | 6 | 113 | 28.2 | full 20-ch model | 0.52, 0.48 | BEV | OK | 4/6 | 60.6 |
| ccs | Ford Focus | ICE | ICE | 4.22 | 1 | 24 | 2.9 | full 20-ch model | 0.31, 0.69 | ICE | OK | 1/1 | 77.8 |
| ccs | Ford Mustang Mach-E | BEV (border) | BEV | 4.84 | 12 | 305 | 34.2 | full 20-ch model | 0.80, 0.20 | BEV | OK | 11/12 | 90.8 |
| ccs | Ford Ranger | ICE | ICE | 5.30 | 3 | 82 | 51.0 | full 20-ch model | 0.33, 0.67 | ICE | OK | 3/3 | 83.3 |
| ccs | Genesis G70 | ICE | ICE | 4.54 | 4 | 128 | 15.0 | full 20-ch model | 0.18, 0.82 | ICE | OK | 4/4 | 97.6 |
| ccs | Genesis G80 | ICE | ICE | 4.95 | 7 | 304 | 45.1 | full 20-ch model | 0.27, 0.73 | ICE | OK | 7/7 | 86.5 |
| ccs | Genesis G90 | ICE | ICE | 5.18 | 9 | 249 | 48.7 | full 20-ch model | 0.27, 0.73 | ICE | OK | 9/9 | 81.6 |
| ccs | Genesis GV60 | BEV | BEV | 4.73 | 2 | 97 | 44.7 | full 20-ch model | 0.78, 0.22 | BEV | OK | 2/2 | 93.0 |
| ccs | Genesis GV70 | ICE | ICE | 4.73 | 8 | 200 | 44.1 | full 20-ch model | 0.21, 0.79 | ICE | OK | 8/8 | 89.2 |
| ccs | Genesis GV80 | ICE | ICE | 4.96 | 3 | 146 | 10.3 | full 20-ch model | 0.11, 0.89 | ICE | OK | 3/3 | 98.9 |
| ccs | Honda Odyssey | ICE | ICE | 5.14 | 96 | 381 | 28.1 | full 20-ch model | 0.19, 0.81 | ICE | OK | 89/96 | 90.9 |
| ccs | Honda Pilot | ICE (border) | ICE | 4.91 | 77 | 396 | 33.9 | full 20-ch model | 0.24, 0.76 | ICE | OK | 71/77 | 86.1 |
| ccs | Honda Ridgeline | ICE | ICE | 5.33 | 56 | 392 | 28.4 | full 20-ch model | 0.19, 0.81 | ICE | OK | 53/56 | 91.6 |
| ccs | Honda e | BEV | BEV | 3.80 | 1 | 2 | 99.1 | full 20-ch model | 0.34, 0.66 | ICE | MISS | 0/1 | 0.0 |
| ccs | Hyundai Custin | ICE (border) | ICE | 4.86 | 4 | 56 | 40.4 | full 20-ch model | 0.16, 0.84 | ICE | OK | 4/4 | 86.8 |
| ccs | Hyundai Elantra | ICE | ICE | 4.31 | 12 | 312 | 40.5 | full 20-ch model | 0.24, 0.76 | ICE | OK | 11/12 | 90.7 |
| ccs | Hyundai Elantra GT / i30 | ICE | ICE | 4.13 | 2 | 100 | 67.7 | full 20-ch model | 0.42, 0.58 | ICE | OK | 2/2 | 66.7 |
| ccs | Hyundai Genesis | ICE (border) | ICE | 4.90 | 13 | 279 | 27.3 | full 20-ch model | 0.23, 0.77 | ICE | OK | 13/13 | 84.8 |
| ccs | Hyundai Ioniq | BEV | BEV | 4.21 | 5 | 235 | 61.9 | full 20-ch model | 0.83, 0.17 | BEV | OK | 5/5 | 90.0 |
| ccs | Hyundai Ioniq 5 | BEV | BEV | 4.83 | 101 | 393 | 46.0 | full 20-ch model | 0.77, 0.23 | BEV | OK | 89/101 | 85.2 |
| ccs | Hyundai Ioniq 6 | BEV | BEV | 4.80 | 24 | 280 | 51.9 | full 20-ch model | 0.74, 0.26 | BEV | OK | 23/24 | 80.3 |
| ccs | Hyundai Kona | BEV | ICE | 4.02 | 20 | 538 | 46.7 | full 20-ch model | 0.75, 0.25 | BEV | OK | 16/20 | 82.4 |
| ccs | Hyundai Palisade | ICE | ICE | 4.96 | 63 | 379 | 41.4 | full 20-ch model | 0.25, 0.75 | ICE | OK | 56/63 | 86.7 |
| ccs | Hyundai Santa Cruz | ICE | ICE | 4.94 | 1 | 49 | 45.0 | full 20-ch model | 0.16, 0.84 | ICE | OK | 1/1 | 93.3 |
| ccs | Hyundai Santa Fe | ICE | ICE | 4.53 | 46 | 1792 | 39.3 | full 20-ch model | 0.27, 0.73 | ICE | OK | 43/46 | 83.4 |
| ccs | Hyundai Sonata | ICE | ICE | 4.62 | 43 | 603 | 48.6 | full 20-ch model | 0.25, 0.75 | ICE | OK | 40/43 | 84.6 |
| ccs | Hyundai Staria | ICE | ICE | 5.65 | 1 | 49 | 31.6 | full 20-ch model | 0.32, 0.69 | ICE | OK | 1/1 | 66.7 |
| ccs | Jeep Grand Cherokee | ICE (border) | ICE | 4.41 | 36 | 1438 | 31.5 | full 20-ch model | 0.15, 0.85 | ICE | OK | 36/36 | 92.8 |
| ccs | Kia Carnival | ICE | ICE | 5.30 | 15 | 477 | 36.6 | full 20-ch model | 0.20, 0.80 | ICE | OK | 15/15 | 92.4 |
| ccs | Kia EV6 | BEV | BEV | 4.74 | 87 | 375 | 46.0 | full 20-ch model | 0.78, 0.22 | BEV | OK | 79/87 | 84.7 |
| ccs | Kia Forte | ICE | ICE | 4.37 | 3 | 76 | 7.7 | full 20-ch model | 0.20, 0.80 | ICE | OK | 3/3 | 87.2 |
| ccs | Kia K5 | ICE | ICE | 4.60 | 3 | 99 | 37.8 | full 20-ch model | 0.23, 0.77 | ICE | OK | 3/3 | 95.5 |
| ccs | Kia Niro | BEV | BEV | 4.27 | 41 | 512 | 33.7 | full 20-ch model | 0.77, 0.23 | BEV | OK | 36/41 | 83.3 |
| ccs | Kia Optima | ICE | ICE | 4.53 | 5 | 163 | 52.7 | full 20-ch model | 0.30, 0.70 | ICE | OK | 5/5 | 69.0 |
| ccs | Kia Sorento | ICE | ICE | 4.64 | 9 | 258 | 36.8 | full 20-ch model | 0.17, 0.83 | ICE | OK | 9/9 | 89.5 |
| ccs | Kia Stinger | ICE | ICE | 4.51 | 10 | 479 | 44.9 | full 20-ch model | 0.30, 0.70 | ICE | OK | 10/10 | 81.3 |
| ccs | Lexus ES | ICE | ICE | 4.61 | 6 | 158 | 27.2 | full 20-ch model | 0.27, 0.73 | ICE | OK | 6/6 | 80.7 |
| ccs | Lexus IS | ICE (border) | ICE | 4.46 | 16 | 558 | 24.9 | full 20-ch model | 0.28, 0.72 | ICE | OK | 14/16 | 79.0 |
| ccs | Lexus LC | ICE | ICE | 4.74 | 2 | 67 | 14.8 | full 20-ch model | 0.12, 0.88 | ICE | OK | 2/2 | 96.2 |
| ccs | Lexus NX | ICE | ICE | 4.20 | 24 | 639 | 34.3 | full 20-ch model | 0.26, 0.74 | ICE | OK | 24/24 | 83.1 |
| ccs | Lexus RC | ICE | ICE | 4.33 | 3 | 52 | 24.5 | full 20-ch model | 0.20, 0.80 | ICE | OK | 2/3 | 86.8 |
| ccs | Lexus RX | ICE | ICE | 4.55 | 69 | 963 | 31.4 | full 20-ch model | 0.30, 0.69 | ICE | OK | 60/69 | 77.8 |
| ccs | Mazda CX-5 | ICE | ICE | 4.31 | 81 | 384 | 40.5 | full 20-ch model | 0.15, 0.85 | ICE | OK | 78/81 | 95.7 |
| ccs | Mazda CX-9 | ICE | ICE | 5.15 | 13 | 362 | 41.8 | full 20-ch model | 0.17, 0.83 | ICE | OK | 13/13 | 91.0 |
| ccs | Nissan Altima | ICE | ICE | 4.56 | 4 | 193 | 35.1 | full 20-ch model | 0.17, 0.83 | ICE | OK | 4/4 | 96.2 |
| ccs | Nissan Leaf | BEV | BEV | 4.16 | 28 | 916 | 34.2 | full 20-ch model | 0.63, 0.37 | BEV | OK | 22/28 | 68.1 |
| ccs | Nissan Rogue | ICE | ICE | 4.31 | 5 | 100 | 63.5 | full 20-ch model | 0.27, 0.73 | ICE | OK | 5/5 | 81.5 |
| ccs | RAM 1500 | ICE | ICE | 6.73 | 99 | 384 | 36.6 | full 20-ch model | 0.17, 0.83 | ICE | OK | 92/99 | 90.8 |
| ccs | RAM 2500/3500 HD | ICE | ICE | 6.57 | 5 | 74 | 19.8 | full 20-ch model | 0.18, 0.82 | ICE | OK | 5/5 | 100.0 |
| ccs | Rivian R1T/R1S | BEV | BEV | 5.58 | 96 | 386 | 44.9 | full 20-ch model | 0.88, 0.12 | BEV | OK | 92/96 | 95.8 |
| ccs | SEAT Ateca | ICE | ICE | 4.11 | 4 | 152 | 50.8 | full 20-ch model | 0.18, 0.82 | ICE | OK | 4/4 | 94.9 |
| ccs | Skoda Kamiq | ICE | ICE | 4.04 | 1 | 35 | 88.1 | full 20-ch model | 0.23, 0.77 | ICE | OK | 1/1 | 100.0 |
| ccs | Skoda Karoq | ICE | ICE | 4.15 | 3 | 8 | 16.3 | full 20-ch model | 0.10, 0.90 | ICE | OK | 3/3 | 100.0 |
| ccs | Skoda Kodiaq | ICE (border) | ICE | 4.41 | 12 | 296 | 32.0 | full 20-ch model | 0.19, 0.81 | ICE | OK | 12/12 | 90.4 |
| ccs | Skoda Octavia | ICE | ICE | 4.15 | 7 | 286 | 39.2 | full 20-ch model | 0.19, 0.81 | ICE | OK | 7/7 | 88.5 |
| ccs | Skoda Superb | ICE | ICE | 4.48 | 10 | 315 | 42.6 | full 20-ch model | 0.19, 0.81 | ICE | OK | 8/10 | 89.7 |
| ccs | Subaru Ascent | ICE | ICE | 4.72 | 34 | 389 | 40.6 | full 20-ch model | 0.20, 0.80 | ICE | OK | 32/34 | 89.7 |
| ccs | Subaru Forester | ICE | ICE | 4.19 | 36 | 437 | 42.0 | full 20-ch model | 0.15, 0.85 | ICE | OK | 35/36 | 95.6 |
| ccs | Subaru Impreza | ICE | ICE | 4.13 | 49 | 709 | 39.3 | full 20-ch model | 0.15, 0.85 | ICE | OK | 48/49 | 93.8 |
| ccs | Subaru Legacy | ICE | ICE | 4.24 | 2 | 51 | 34.0 | full 20-ch model | 0.14, 0.86 | ICE | OK | 2/2 | 95.7 |
| ccs | Subaru Outback | ICE | ICE | 4.23 | 46 | 491 | 40.5 | full 20-ch model | 0.16, 0.84 | ICE | OK | 46/46 | 93.8 |
| ccs | Tesla Model 3 | BEV | BEV | 4.57 | 28 | 352 | 52.9 | full 20-ch model | 0.80, 0.20 | BEV | OK | 27/28 | 83.6 |
| ccs | Tesla Model Y | BEV | BEV | 4.73 | 21 | 538 | 42.5 | full 20-ch model | 0.84, 0.16 | BEV | OK | 20/21 | 89.3 |
| ccs | Toyota Avalon | ICE (border) | ICE | 4.47 | 6 | 239 | 15.1 | full 20-ch model | 0.18, 0.82 | ICE | OK | 6/6 | 89.8 |
| ccs | Toyota Corolla | ICE | ICE | 4.14 | 39 | 981 | 36.8 | full 20-ch model | 0.19, 0.81 | ICE | OK | 39/39 | 91.8 |
| ccs | Toyota Highlander | ICE | ICE | 4.60 | 225 | 765 | 38.4 | full 20-ch model | 0.36, 0.64 | ICE | OK | 148/225 | 66.2 |
| ccs | Toyota RAV4 | ICE | ICE | 4.34 | 307 | 1165 | 43.2 | full 20-ch model | 0.32, 0.68 | ICE | OK | 224/307 | 78.1 |
| ccs | Toyota Sienna | ICE | ICE | 5.21 | 62 | 975 | 30.5 | full 20-ch model | 0.25, 0.75 | ICE | OK | 58/62 | 87.6 |
| ccs | Volkswagen Atlas | ICE | ICE | 5.11 | 32 | 961 | 38.0 | full 20-ch model | 0.14, 0.86 | ICE | OK | 32/32 | 94.2 |
| ccs | Volkswagen Jetta | ICE | ICE | 4.17 | 12 | 342 | 27.0 | full 20-ch model | 0.12, 0.88 | ICE | OK | 12/12 | 94.5 |
| ccs | Volkswagen T-Roc | ICE | ICE | 3.99 | 3 | 147 | 24.7 | full 20-ch model | 0.16, 0.84 | ICE | OK | 3/3 | 91.5 |
| ccs | Volkswagen Taos | ICE | ICE | 4.18 | 2 | 55 | 30.4 | full 20-ch model | 0.17, 0.83 | ICE | OK | 2/2 | 95.8 |
| ccs | Volkswagen Touran | ICE (border) | ICE | 4.39 | 1 | 44 | 9.2 | full 20-ch model | 0.23, 0.77 | ICE | OK | 1/1 | 92.9 |
| ccs | Volkswagen Transporter | ICE (border) | ICE | 4.91 | 2 | 74 | 30.4 | full 20-ch model | 0.28, 0.72 | ICE | OK | 2/2 | 80.0 |
| ccs_honda | Honda CR-V | ICE | ICE | 4.28 | 96 | 486 | 45.0 | full 20-ch model | 0.18, 0.82 | ICE | OK | 95/96 | 95.0 |
| ccs_honda | Honda Civic | ICE | ICE | 4.20 | 123 | 1173 | 43.7 | full 20-ch model | 0.17, 0.83 | ICE | OK | 121/123 | 93.6 |
| ccs_honda | Honda Fit | ICE | ICE | 3.74 | 7 | 273 | 35.0 | full 20-ch model | 0.17, 0.83 | ICE | OK | 7/7 | 94.0 |
| ccs_honda | Honda HR-V | ICE | ICE | 4.01 | 5 | 152 | 33.3 | full 20-ch model | 0.17, 0.83 | ICE | OK | 5/5 | 94.9 |
| comma2k19 | Honda Civic | ICE | ICE | 4.20 | 1 | 196 | 61.7 | full 20-ch model | 0.29, 0.71 | ICE | OK | 1/1 | 90.5 |
| comma2k19 | Toyota RAV4 | ICE | ICE | 4.13 | 1 | 185 | 55.7 | full 20-ch model | 0.32, 0.68 | ICE | OK | 1/1 | 85.7 |

### size: every named external car model, primary run of its dataset (mean probability over all windows of the model -> arg-max)

| dataset | car model | label | powertrain | footprint | veh | win | DAS % | model | P(SMALL/MID/LARGE) | decision | hit | vehicles correct | drive acc % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ultraav | Audi A4 Avant | SMALL (border) | HEV | 4.41 | 1 | 154 | 95.5 | V3 variant | 0.58, 0.35, 0.07 | SMALL | OK | 1/1 | 100.0 |
| ultraav | Audi A6 | MID | ICE | 4.75 | 1 | 173 | 100.0 | V3 variant | 0.66, 0.27, 0.07 | SMALL | MISS | 0/1 | 0.0 |
| ultraav | Audi e-tron | MID (border) | BEV | 4.84 | 1 | 219 | 95.4 | V3 variant | 0.52, 0.35, 0.13 | SMALL | MISS | 0/1 | 6.0 |
| ultraav | BMW X5 | LARGE | ICE | 4.99 | 1 | 173 | 100.0 | V3 variant | 0.68, 0.27, 0.06 | SMALL | MISS | 0/1 | 0.0 |
| ultraav | BMW i3s REx | SMALL | PHEV | 4.07 | 1 | 236 | 96.2 | V3 variant | 0.53, 0.34, 0.13 | SMALL | OK | 1/1 | 93.5 |
| ultraav | Ford Fusion | MID | unknown | 4.53 | 1 | 1 | 0.0 | V3 variant | 0.91, 0.06, 0.03 | SMALL | MISS | 0/1 | 0.0 |
| ultraav | Ford S-Max | MID | ICE | 4.57 | 1 | 59 | 100.0 | V3 variant | 0.62, 0.18, 0.20 | SMALL | MISS | 0/1 | 0.0 |
| ultraav | Hyundai Nexo | MID | FCEV | 4.52 | 1 | 51 | 100.0 | V3 variant | 0.70, 0.24, 0.06 | SMALL | MISS | 0/1 | 0.0 |
| ultraav | Jaguar I-Pace | LARGE (border) | BEV | 4.92 | 1 | 195 | 95.9 | V3 variant | 0.59, 0.33, 0.08 | SMALL | MISS | 0/1 | 0.0 |
| ultraav | Kia Niro PHEV | SMALL | PHEV | 4.29 | 1 | 64 | 100.0 | V3 variant | 0.70, 0.13, 0.17 | SMALL | OK | 1/1 | 100.0 |
| ultraav | Lincoln MKZ | MID | unknown | 4.50 | 2 | 53 | 100.0 | V3 variant | 0.46, 0.39, 0.15 | SMALL | MISS | 0/2 | 7.4 |
| ultraav | Mazda 3 | SMALL | ICE | 4.20 | 1 | 79 | 100.0 | V3 variant | 0.52, 0.34, 0.14 | SMALL | OK | 1/1 | 80.0 |
| ultraav | Mercedes-Benz A-Class | SMALL | ICE | 4.25 | 1 | 173 | 100.0 | V3 variant | 0.67, 0.28, 0.06 | SMALL | OK | 1/1 | 100.0 |
| ultraav | Mercedes-Benz GLE 450 4MATIC | LARGE | HEV | 5.02 | 1 | 272 | 96.0 | V3 variant | 0.60, 0.32, 0.08 | SMALL | MISS | 0/1 | 1.5 |
| ultraav | Mitsubishi Outlander PHEV | SMALL | PHEV | 4.11 | 1 | 53 | 100.0 | V3 variant | 0.64, 0.14, 0.22 | SMALL | OK | 1/1 | 100.0 |
| ultraav | Tesla Model 3 | MID | BEV | 4.57 | 2 | 388 | 96.9 | V3 variant | 0.60, 0.31, 0.09 | SMALL | MISS | 0/2 | 3.8 |
| ultraav | Tesla Model S | LARGE | BEV | 4.97 | 1 | 255 | 95.7 | V3 variant | 0.58, 0.34, 0.08 | SMALL | MISS | 0/1 | 0.0 |
| ultraav | Tesla Model X | LARGE | BEV | 5.06 | 1 | 218 | 98.2 | V3 variant | 0.55, 0.33, 0.12 | SMALL | MISS | 0/1 | 4.0 |
| ultraav | Toyota RAV4 Hybrid | SMALL | HEV | 4.34 | 1 | 105 | 100.0 | V3 variant | 0.49, 0.33, 0.18 | SMALL | OK | 1/1 | 77.8 |
| ultraav | Volkswagen e-Golf | SMALL | BEV | 4.02 | 1 | 55 | 100.0 | V3 variant | 0.78, 0.12, 0.10 | SMALL | OK | 1/1 | 100.0 |
| openacc | Audi A4 Avant | SMALL (border) | HEV | 4.41 | 1 | 225 | 92.4 | V3 variant | 0.56, 0.36, 0.08 | SMALL | OK | 1/1 | 97.9 |
| openacc | Audi A6 | MID | ICE | 4.75 | 1 | 220 | 79.1 | V3 variant | 0.65, 0.27, 0.08 | SMALL | MISS | 0/1 | 0.0 |
| openacc | Audi A8 | LARGE (border) | ICE | 4.91 | 1 | 220 | 100.0 | V3 variant | 0.64, 0.25, 0.12 | SMALL | MISS | 0/1 | 0.0 |
| openacc | Audi e-tron | MID (border) | BEV | 4.84 | 1 | 350 | 90.3 | V3 variant | 0.51, 0.35, 0.13 | SMALL | MISS | 0/1 | 13.4 |
| openacc | BMW X5 | LARGE | ICE | 4.99 | 1 | 220 | 79.1 | V3 variant | 0.66, 0.27, 0.07 | SMALL | MISS | 0/1 | 0.0 |
| openacc | Fiat 500X | SMALL | ICE | 3.97 | 1 | 182 | 0.0 | V3 variant | 0.64, 0.16, 0.20 | SMALL | OK | 1/1 | 83.3 |
| openacc | Ford S-Max | MID | ICE | 4.57 | 1 | 217 | 83.1 | V3 variant | 0.53, 0.27, 0.20 | SMALL | MISS | 0/1 | 11.1 |
| openacc | Jaguar I-Pace | LARGE (border) | BEV | 4.92 | 1 | 292 | 88.7 | V3 variant | 0.57, 0.33, 0.09 | SMALL | MISS | 0/1 | 0.0 |
| openacc | MINI Cooper | SMALL | ICE | 3.74 | 1 | 42 | 0.0 | V3 variant | 0.52, 0.23, 0.25 | SMALL | OK | 1/1 | 100.0 |
| openacc | Mazda 3 | SMALL | ICE | 4.29 | 1 | 305 | 88.9 | V3 variant | 0.51, 0.34, 0.15 | SMALL | OK | 1/1 | 93.7 |
| openacc | Mercedes-Benz A-Class | SMALL | ICE | 4.25 | 1 | 220 | 79.1 | V3 variant | 0.65, 0.28, 0.07 | SMALL | OK | 1/1 | 100.0 |
| openacc | Mercedes-Benz GLE 450 4MATIC | LARGE | HEV | 5.02 | 1 | 360 | 91.1 | V3 variant | 0.59, 0.33, 0.08 | SMALL | MISS | 0/1 | 0.0 |
| openacc | Mitsubishi Space Star | SMALL | ICE | 3.48 | 1 | 114 | 0.0 | V3 variant | 0.64, 0.17, 0.20 | SMALL | OK | 1/1 | 100.0 |
| openacc | SsangYong Rexton | MID | ICE | 4.70 | 1 | 69 | 0.0 | V3 variant | 0.62, 0.31, 0.07 | SMALL | MISS | 0/1 | 0.0 |
| openacc | Tesla Model 3 | MID | BEV | 4.57 | 2 | 486 | 84.4 | V3 variant | 0.59, 0.32, 0.09 | SMALL | MISS | 0/2 | 6.9 |
| openacc | Tesla Model S | LARGE | BEV | 4.97 | 1 | 334 | 89.8 | V3 variant | 0.55, 0.35, 0.10 | SMALL | MISS | 0/1 | 1.4 |
| openacc | Tesla Model X | LARGE | BEV | 5.06 | 1 | 345 | 90.4 | V3 variant | 0.53, 0.35, 0.12 | SMALL | MISS | 0/1 | 5.8 |
| openacc | Volkswagen Polo | SMALL | ICE | 3.61 | 1 | 82 | 0.0 | V3 variant | 0.25, 0.30, 0.45 | LARGE | MISS | 0/1 | 0.0 |
| openacc | Volkswagen e-Golf | SMALL | BEV | 4.02 | 1 | 73 | 90.1 | V3 variant | 0.71, 0.17, 0.12 | SMALL | OK | 1/1 | 100.0 |
| openacc | Volvo XC40 | SMALL | ICE | 4.36 | 1 | 184 | 86.5 | V3 variant | 0.72, 0.14, 0.14 | SMALL | OK | 1/1 | 83.3 |
| microsim_ev | Hyundai Ioniq 5 | MID (border) | BEV | 4.83 | 1 | 99 | 100.0 | V3 variant | 0.48, 0.38, 0.14 | SMALL | MISS | 0/1 | 21.4 |
| microsim_ev | Polestar 2 | SMALL (border) | BEV | 4.38 | 1 | 8 | 100.0 | V3 variant | 0.60, 0.35, 0.04 | SMALL | OK | 1/1 | 100.0 |
| microsim_ev | Tesla Model 3 | MID | BEV | 4.57 | 1 | 4 | 100.0 | V3 variant | 0.47, 0.41, 0.12 | SMALL | MISS | 0/1 | 0.0 |
| uga_ev | Ford Mustang Mach-E | MID (border) | BEV | 4.84 | 1 | 83 | 67.5 | V3 variant | 0.55, 0.37, 0.08 | SMALL | MISS | 0/1 | 0.0 |
| uga_ev | Kia Niro EV | SMALL | BEV | 4.26 | 2 | 161 | 67.1 | V3 variant | 0.61, 0.30, 0.09 | SMALL | OK | 2/2 | 100.0 |
| uga_ev | Lexus UX 200 | SMALL | ICE | 4.09 | 1 | 82 | 0.0 | V3 variant | 0.60, 0.34, 0.06 | SMALL | OK | 1/1 | 100.0 |
| tesla_yuan | Tesla Model 3 | MID | BEV | 4.57 | 1 | 2619 | 0.0 | V3 variant | 0.19, 0.50, 0.31 | MID | OK | 1/1 | 78.3 |
| openacc_500x | Fiat 500X | SMALL | ICE | 3.97 | 1 | 13523 | 0.0 | V3 variant | 0.47, 0.39, 0.14 | SMALL | OK | 1/1 | 46.2 |
| ved | Nissan Leaf (2013) | SMALL | BEV | 4.15 | 3 | 431 | 0.0 | V3 variant | 0.47, 0.39, 0.15 | SMALL | OK | 3/3 | 62.5 |
| a2d2 | Audi Q7 e-tron | LARGE | PHEV | 5.05 | 1 | 34 | 0.0 | full 20-ch model | 0.85, 0.09, 0.06 | SMALL | MISS | 0/1 | 0.0 |
| ccs_sample | Chevrolet Bolt EUV | SMALL | BEV | 3.99 | 1 | 1 | 35.3 | full 20-ch model | 0.15, 0.71, 0.14 | MID | MISS | 0/1 | 0.0 |
| ccs_sample | Rivian R1T/R1S | LARGE | BEV | 5.58 | 1 | 1 | 99.8 | full 20-ch model | 0.24, 0.43, 0.33 | MID | MISS | 0/1 | 0.0 |
| ccs_sample | Tesla Model 3 | MID | BEV | 4.57 | 1 | 1 | 0.0 | full 20-ch model | 0.21, 0.65, 0.14 | MID | OK | 1/1 | 100.0 |
| ccs_sample | Toyota RAV4 | SMALL | ICE | 4.34 | 1 | 1 | 99.8 | full 20-ch model | 0.86, 0.12, 0.02 | SMALL | OK | 1/1 | 100.0 |
| csc | Chevrolet Bolt EUV | SMALL | BEV | 3.99 | 1 | 1500 | 90.7 | V1 variant | 0.54, 0.30, 0.16 | SMALL | OK | 1/1 | 70.5 |
| csc | Chevrolet Silverado 1500 | LARGE | ICE | 6.65 | 1 | 1500 | 92.4 | V1 variant | 0.28, 0.34, 0.38 | LARGE | OK | 1/1 | 37.2 |
| csc | Genesis GV60 | MID | BEV | 4.73 | 1 | 999 | 94.9 | V1 variant | 0.38, 0.43, 0.19 | MID | OK | 1/1 | 51.2 |
| csc | Honda Accord | MID | ICE | 4.54 | 1 | 1500 | 91.9 | V1 variant | 0.52, 0.35, 0.13 | SMALL | MISS | 0/1 | 27.4 |
| csc | Honda CR-V | SMALL | ICE | 4.23 | 2 | 2500 | 90.8 | V1 variant | 0.48, 0.22, 0.29 | SMALL | OK | 1/2 | 63.3 |
| csc | Honda Civic | SMALL | ICE | 4.20 | 3 | 4500 | 91.3 | V1 variant | 0.58, 0.36, 0.06 | SMALL | OK | 3/3 | 78.1 |
| csc | Honda Odyssey | LARGE | ICE | 5.14 | 1 | 1500 | 91.6 | V1 variant | 0.43, 0.34, 0.23 | SMALL | MISS | 0/1 | 21.7 |
| csc | Honda Pilot | LARGE (border) | ICE | 4.91 | 1 | 1500 | 92.9 | V1 variant | 0.28, 0.20, 0.52 | LARGE | OK | 1/1 | 58.5 |
| csc | Hyundai Elantra | SMALL | ICE | 4.31 | 1 | 1500 | 90.6 | V1 variant | 0.56, 0.36, 0.07 | SMALL | OK | 1/1 | 72.7 |
| csc | Hyundai Ioniq | SMALL | BEV | 4.22 | 1 | 497 | 92.8 | V1 variant | 0.40, 0.51, 0.09 | MID | MISS | 0/1 | 34.8 |
| csc | Hyundai Ioniq 5 | MID (border) | BEV | 4.83 | 1 | 1500 | 92.3 | V1 variant | 0.37, 0.43, 0.20 | MID | OK | 1/1 | 49.3 |
| csc | Hyundai Kona | SMALL | BEV | 4.02 | 1 | 1500 | 92.6 | V1 variant | 0.56, 0.35, 0.10 | SMALL | OK | 1/1 | 72.9 |
| csc | Hyundai Palisade | LARGE | ICE | 4.96 | 1 | 1500 | 93.3 | V1 variant | 0.45, 0.36, 0.19 | SMALL | MISS | 0/1 | 13.6 |
| csc | Hyundai Santa Fe | MID | ICE | 4.53 | 2 | 3000 | 92.9 | V1 variant | 0.51, 0.28, 0.21 | SMALL | MISS | 0/2 | 19.4 |
| csc | Hyundai Sonata | MID | ICE | 4.56 | 2 | 3000 | 90.7 | V1 variant | 0.53, 0.32, 0.15 | SMALL | MISS | 0/2 | 20.0 |
| csc | Hyundai Tucson | MID | ICE | 4.53 | 1 | 999 | 93.2 | V1 variant | 0.39, 0.45, 0.16 | MID | OK | 1/1 | 50.4 |
| csc | Kia EV6 | MID | BEV | 4.74 | 1 | 1500 | 93.1 | V1 variant | 0.55, 0.36, 0.10 | SMALL | MISS | 0/1 | 26.9 |
| csc | Kia Niro | SMALL | BEV | 4.27 | 1 | 1500 | 90.4 | V1 variant | 0.56, 0.35, 0.09 | SMALL | OK | 1/1 | 71.7 |
| csc | Nissan Leaf | SMALL | BEV | 4.16 | 2 | 2967 | 91.0 | V1 variant | 0.53, 0.31, 0.16 | SMALL | OK | 2/2 | 61.2 |
| csc | RAM 1500 | LARGE | ICE | 6.73 | 1 | 1500 | 94.1 | V1 variant | 0.36, 0.25, 0.39 | LARGE | OK | 1/1 | 38.9 |
| csc | Toyota Camry | MID (border) | ICE | 4.47 | 2 | 3000 | 92.2 | V1 variant | 0.58, 0.34, 0.08 | SMALL | MISS | 0/2 | 21.9 |
| csc | Toyota Corolla | SMALL | ICE | 4.14 | 2 | 3000 | 91.4 | V1 variant | 0.50, 0.32, 0.18 | SMALL | OK | 2/2 | 60.6 |
| csc | Toyota Highlander | MID | ICE | 4.60 | 2 | 3000 | 93.1 | V1 variant | 0.42, 0.36, 0.23 | SMALL | MISS | 1/2 | 33.6 |
| csc | Toyota RAV4 | SMALL | ICE | 4.13 | 3 | 4500 | 92.6 | V1 variant | 0.53, 0.32, 0.15 | SMALL | OK | 3/3 | 69.0 |
| csc | Toyota Sienna | LARGE | ICE | 5.21 | 1 | 1500 | 93.1 | V1 variant | 0.34, 0.27, 0.39 | LARGE | OK | 1/1 | 40.9 |
| moretorque | Hyundai Sonata | MID | ICE | 4.62 | 1 | 2926 | 63.4 | full 20-ch model | 0.52, 0.23, 0.25 | SMALL | MISS | 0/1 | 13.8 |
| moretorque | Mazda 3 | SMALL | ICE | 4.29 | 7 | 2319 | 72.8 | full 20-ch model | 0.75, 0.18, 0.07 | SMALL | OK | 7/7 | 91.7 |
| moretorque | Mazda CX-50 | MID | ICE | 4.67 | 3 | 1071 | 14.0 | full 20-ch model | 0.64, 0.22, 0.14 | SMALL | MISS | 0/3 | 6.4 |
| moretorque | Omoda E5 | SMALL | BEV | 4.05 | 1 | 874 | 29.6 | full 20-ch model | 0.55, 0.24, 0.20 | SMALL | OK | 1/1 | 57.3 |
| ccs | Acura ILX | SMALL | ICE | 4.06 | 1 | 4 | 29.8 | full 20-ch model | 0.40, 0.29, 0.30 | SMALL | OK | 1/1 | 66.7 |
| ccs | Acura RDX | MID | ICE | 4.50 | 21 | 476 | 35.2 | full 20-ch model | 0.63, 0.33, 0.04 | SMALL | MISS | 0/21 | 14.4 |
| ccs | Audi A3 | SMALL | unknown | 3.97 | 6 | 237 | 34.8 | full 20-ch model | 0.73, 0.24, 0.03 | SMALL | OK | 5/6 | 95.3 |
| ccs | Audi Q3 | SMALL | ICE | 4.23 | 11 | 380 | 17.8 | full 20-ch model | 0.72, 0.25, 0.03 | SMALL | OK | 11/11 | 89.1 |
| ccs | Buick Regal Sportback | MID | ICE | 4.52 | 1 | 50 | 48.9 | full 20-ch model | 0.61, 0.32, 0.07 | SMALL | MISS | 0/1 | 18.2 |
| ccs | Cadillac Escalade | LARGE | ICE | 5.14 | 1 | 1 | 99.8 | full 20-ch model | 0.65, 0.16, 0.19 | SMALL | MISS | 0/1 | 0.0 |
| ccs | Cadillac Escalade ESV | LARGE | ICE | 5.76 | 1 | 8 | 55.5 | full 20-ch model | 0.69, 0.23, 0.08 | SMALL | MISS | 0/1 | 0.0 |
| ccs | Cadillac XT4 | MID (border) | ICE | 4.46 | 1 | 1 | 1.2 | full 20-ch model | 0.79, 0.21, 0.01 | SMALL | MISS | 0/1 | 0.0 |
| ccs | Chevrolet Bolt EUV | SMALL | BEV | 3.99 | 160 | 380 | 43.3 | full 20-ch model | 0.67, 0.23, 0.10 | SMALL | OK | 142/160 | 88.9 |
| ccs | Chevrolet Equinox | SMALL | ICE | 4.30 | 5 | 243 | 45.7 | full 20-ch model | 0.64, 0.31, 0.05 | SMALL | OK | 5/5 | 77.6 |
| ccs | Chevrolet Silverado 1500 | LARGE | ICE | 6.65 | 52 | 395 | 48.3 | full 20-ch model | 0.36, 0.23, 0.42 | LARGE | OK | 31/52 | 53.3 |
| ccs | Chevrolet Traverse | LARGE | ICE | 5.24 | 1 | 29 | 47.5 | full 20-ch model | 0.42, 0.12, 0.46 | LARGE | OK | 1/1 | 70.6 |
| ccs | Chevrolet Volt | SMALL | PHEV | 4.18 | 25 | 391 | 35.2 | full 20-ch model | 0.70, 0.23, 0.06 | SMALL | OK | 24/25 | 88.7 |
| ccs | Chrysler Pacifica | LARGE | PHEV | 5.36 | 184 | 1475 | 30.8 | full 20-ch model | 0.50, 0.30, 0.21 | SMALL | MISS | 27/184 | 20.8 |
| ccs | Dodge Durango | LARGE | ICE | 4.95 | 1 | 11 | 0.0 | full 20-ch model | 0.59, 0.20, 0.22 | SMALL | MISS | 0/1 | 0.0 |
| ccs | Ford Bronco Sport | SMALL | ICE | 4.28 | 8 | 94 | 60.5 | full 20-ch model | 0.42, 0.50, 0.08 | MID | MISS | 3/8 | 46.7 |
| ccs | Ford Escape | SMALL (border) | unknown | 4.39 | 14 | 270 | 42.2 | full 20-ch model | 0.57, 0.34, 0.09 | SMALL | OK | 12/14 | 67.8 |
| ccs | Ford Explorer | LARGE | unknown | 5.14 | 44 | 374 | 30.6 | full 20-ch model | 0.49, 0.30, 0.21 | SMALL | MISS | 5/44 | 22.0 |
| ccs | Ford F-150 | LARGE | unknown | 6.44 | 42 | 387 | 45.8 | full 20-ch model | 0.31, 0.25, 0.44 | LARGE | OK | 25/42 | 60.3 |
| ccs | Ford F-150 Lightning | LARGE | BEV | 6.41 | 6 | 113 | 28.2 | full 20-ch model | 0.26, 0.20, 0.55 | LARGE | OK | 5/6 | 84.8 |
| ccs | Ford Focus | SMALL | ICE | 4.22 | 1 | 24 | 2.9 | full 20-ch model | 0.72, 0.24, 0.03 | SMALL | OK | 1/1 | 88.9 |
| ccs | Ford Maverick | LARGE (border) | unknown | 4.93 | 32 | 382 | 52.3 | full 20-ch model | 0.47, 0.32, 0.21 | SMALL | MISS | 2/32 | 20.3 |
| ccs | Ford Mustang Mach-E | MID (border) | BEV | 4.84 | 12 | 305 | 34.2 | full 20-ch model | 0.40, 0.46, 0.13 | MID | OK | 7/12 | 59.2 |
| ccs | Ford Ranger | LARGE | ICE | 5.30 | 3 | 82 | 51.0 | full 20-ch model | 0.43, 0.19, 0.38 | SMALL | MISS | 2/3 | 63.9 |
| ccs | Genesis G70 | MID | ICE | 4.54 | 4 | 128 | 15.0 | full 20-ch model | 0.66, 0.31, 0.03 | SMALL | MISS | 0/4 | 43.9 |
| ccs | Genesis G80 | LARGE | ICE | 4.95 | 7 | 304 | 45.1 | full 20-ch model | 0.38, 0.36, 0.26 | SMALL | MISS | 1/7 | 27.0 |
| ccs | Genesis G90 | LARGE | ICE | 5.18 | 9 | 249 | 48.7 | full 20-ch model | 0.53, 0.41, 0.06 | SMALL | MISS | 0/9 | 2.4 |
| ccs | Genesis GV60 | MID | BEV | 4.73 | 2 | 97 | 44.7 | full 20-ch model | 0.35, 0.52, 0.13 | MID | OK | 1/2 | 48.8 |
| ccs | Genesis GV70 | MID | ICE | 4.73 | 8 | 200 | 44.1 | full 20-ch model | 0.57, 0.36, 0.07 | SMALL | MISS | 1/8 | 21.6 |
| ccs | Genesis GV80 | LARGE | ICE | 4.96 | 3 | 146 | 10.3 | full 20-ch model | 0.72, 0.22, 0.06 | SMALL | MISS | 0/3 | 2.2 |
| ccs | Honda Accord | MID | HEV | 4.54 | 120 | 390 | 49.0 | full 20-ch model | 0.49, 0.45, 0.07 | SMALL | MISS | 55/120 | 45.0 |
| ccs | Honda CR-V | SMALL | HEV | 4.28 | 13 | 304 | 41.8 | full 20-ch model | 0.68, 0.27, 0.05 | SMALL | OK | 13/13 | 89.7 |
| ccs | Honda Insight | SMALL | HEV | 4.20 | 15 | 334 | 51.8 | full 20-ch model | 0.71, 0.26, 0.03 | SMALL | OK | 14/15 | 92.3 |
| ccs | Honda Odyssey | LARGE | ICE | 5.14 | 96 | 381 | 28.1 | full 20-ch model | 0.56, 0.36, 0.09 | SMALL | MISS | 2/96 | 2.8 |
| ccs | Honda Pilot | LARGE (border) | ICE | 4.91 | 77 | 396 | 33.9 | full 20-ch model | 0.50, 0.41, 0.09 | SMALL | MISS | 2/77 | 4.2 |
| ccs | Honda Ridgeline | LARGE | ICE | 5.33 | 56 | 392 | 28.4 | full 20-ch model | 0.47, 0.36, 0.17 | SMALL | MISS | 2/56 | 11.7 |
| ccs | Honda e | SMALL | BEV | 3.80 | 1 | 2 | 99.1 | full 20-ch model | 0.75, 0.24, 0.01 | SMALL | OK | 1/1 | 100.0 |
| ccs | Hyundai Azera | MID | HEV | 4.63 | 1 | 5 | 79.9 | full 20-ch model | 0.63, 0.32, 0.06 | SMALL | MISS | 0/1 | 40.0 |
| ccs | Hyundai Custin | MID (border) | ICE | 4.86 | 4 | 56 | 40.4 | full 20-ch model | 0.29, 0.24, 0.47 | LARGE | MISS | 0/4 | 21.1 |
| ccs | Hyundai Elantra | SMALL | HEV | 4.31 | 23 | 646 | 41.2 | full 20-ch model | 0.65, 0.32, 0.04 | SMALL | OK | 23/23 | 82.4 |
| ccs | Hyundai Elantra GT / i30 | SMALL | ICE | 4.13 | 2 | 100 | 67.7 | full 20-ch model | 0.69, 0.28, 0.03 | SMALL | OK | 2/2 | 88.2 |
| ccs | Hyundai Genesis | LARGE (border) | ICE | 4.90 | 13 | 279 | 27.3 | full 20-ch model | 0.35, 0.41, 0.24 | MID | MISS | 2/13 | 20.6 |
| ccs | Hyundai Ioniq | SMALL | PHEV | 4.22 | 26 | 935 | 49.1 | full 20-ch model | 0.70, 0.27, 0.03 | SMALL | OK | 26/26 | 87.1 |
| ccs | Hyundai Ioniq 5 | MID | BEV | 4.83 | 101 | 393 | 46.0 | full 20-ch model | 0.47, 0.49, 0.04 | MID | OK | 46/101 | 46.7 |
| ccs | Hyundai Ioniq 6 | MID | BEV | 4.80 | 24 | 280 | 51.9 | full 20-ch model | 0.53, 0.43, 0.04 | SMALL | MISS | 7/24 | 37.6 |
| ccs | Hyundai Ioniq Plug-in | SMALL | PHEV | 4.22 | 2 | 100 | 43.2 | full 20-ch model | 0.70, 0.23, 0.07 | SMALL | OK | 2/2 | 88.6 |
| ccs | Hyundai Kona | SMALL | ICE | 4.02 | 20 | 538 | 46.7 | full 20-ch model | 0.63, 0.35, 0.03 | SMALL | OK | 18/20 | 77.3 |
| ccs | Hyundai Palisade | LARGE | ICE | 4.96 | 63 | 379 | 41.4 | full 20-ch model | 0.35, 0.34, 0.30 | SMALL | MISS | 24/63 | 34.1 |
| ccs | Hyundai Santa Cruz | LARGE | ICE | 4.94 | 1 | 49 | 45.0 | full 20-ch model | 0.58, 0.32, 0.09 | SMALL | MISS | 0/1 | 6.7 |
| ccs | Hyundai Santa Fe | MID | ICE | 4.53 | 62 | 2355 | 40.4 | full 20-ch model | 0.44, 0.46, 0.10 | MID | OK | 39/62 | 48.3 |
| ccs | Hyundai Sonata | MID | ICE | 4.62 | 56 | 931 | 46.9 | full 20-ch model | 0.61, 0.33, 0.06 | SMALL | MISS | 4/56 | 22.4 |
| ccs | Hyundai Staria | LARGE | ICE | 5.65 | 1 | 49 | 31.6 | full 20-ch model | 0.64, 0.31, 0.04 | SMALL | MISS | 0/1 | 0.0 |
| ccs | Hyundai Tucson | MID | unknown | 4.53 | 27 | 321 | 41.7 | full 20-ch model | 0.64, 0.31, 0.04 | SMALL | MISS | 3/27 | 17.8 |
| ccs | Jeep Grand Cherokee | SMALL (border) | ICE | 4.41 | 36 | 1438 | 31.5 | full 20-ch model | 0.48, 0.39, 0.12 | SMALL | OK | 26/36 | 55.2 |
| ccs | Kia Carnival | LARGE | ICE | 5.30 | 15 | 477 | 36.6 | full 20-ch model | 0.55, 0.38, 0.07 | SMALL | MISS | 0/15 | 2.7 |
| ccs | Kia EV6 | MID | BEV | 4.74 | 87 | 375 | 46.0 | full 20-ch model | 0.30, 0.63, 0.07 | MID | OK | 70/87 | 74.5 |
| ccs | Kia Forte | SMALL | ICE | 4.37 | 3 | 76 | 7.7 | full 20-ch model | 0.72, 0.25, 0.04 | SMALL | OK | 3/3 | 76.9 |
| ccs | Kia K5 | MID | ICE | 4.60 | 3 | 99 | 37.8 | full 20-ch model | 0.71, 0.26, 0.03 | SMALL | MISS | 0/3 | 7.5 |
| ccs | Kia Niro | SMALL | HEV | 4.29 | 57 | 1001 | 42.8 | full 20-ch model | 0.68, 0.29, 0.03 | SMALL | OK | 53/57 | 86.0 |
| ccs | Kia Optima | MID | ICE | 4.53 | 5 | 163 | 52.7 | full 20-ch model | 0.67, 0.30, 0.03 | SMALL | MISS | 0/5 | 19.7 |
| ccs | Kia Sorento | MID | ICE | 4.64 | 17 | 545 | 28.8 | full 20-ch model | 0.69, 0.26, 0.05 | SMALL | MISS | 0/17 | 9.2 |
| ccs | Kia Sportage | SMALL (border) | unknown | 4.42 | 9 | 277 | 39.3 | full 20-ch model | 0.65, 0.31, 0.04 | SMALL | OK | 9/9 | 82.1 |
| ccs | Kia Stinger | MID | ICE | 4.51 | 10 | 479 | 44.9 | full 20-ch model | 0.36, 0.56, 0.09 | MID | OK | 10/10 | 61.0 |
| ccs | Lexus CT 200h | SMALL | HEV | 3.98 | 3 | 65 | 23.8 | full 20-ch model | 0.55, 0.34, 0.11 | SMALL | OK | 3/3 | 64.3 |
| ccs | Lexus ES | MID | HEV | 4.61 | 68 | 533 | 37.7 | full 20-ch model | 0.54, 0.41, 0.06 | SMALL | MISS | 21/68 | 31.8 |
| ccs | Lexus IS | SMALL (border) | ICE | 4.46 | 16 | 558 | 24.9 | full 20-ch model | 0.63, 0.32, 0.05 | SMALL | OK | 16/16 | 80.5 |
| ccs | Lexus LC | MID | ICE | 4.74 | 2 | 67 | 14.8 | full 20-ch model | 0.68, 0.29, 0.04 | SMALL | MISS | 0/2 | 9.4 |
| ccs | Lexus NX | SMALL | ICE | 4.20 | 24 | 639 | 34.3 | full 20-ch model | 0.70, 0.28, 0.02 | SMALL | OK | 23/24 | 88.7 |
| ccs | Lexus RC | SMALL | ICE | 4.33 | 3 | 52 | 24.5 | full 20-ch model | 0.66, 0.30, 0.03 | SMALL | OK | 3/3 | 92.1 |
| ccs | Lexus RX | MID | ICE | 4.55 | 124 | 1931 | 30.2 | full 20-ch model | 0.64, 0.32, 0.04 | SMALL | MISS | 7/124 | 20.4 |
| ccs | Mazda CX-5 | SMALL | ICE | 4.31 | 81 | 384 | 40.5 | full 20-ch model | 0.71, 0.25, 0.04 | SMALL | OK | 79/81 | 94.2 |
| ccs | Mazda CX-9 | LARGE | ICE | 5.15 | 13 | 362 | 41.8 | full 20-ch model | 0.48, 0.16, 0.35 | SMALL | MISS | 4/13 | 43.9 |
| ccs | Nissan Altima | MID | ICE | 4.56 | 4 | 193 | 35.1 | full 20-ch model | 0.58, 0.32, 0.10 | SMALL | MISS | 0/4 | 17.1 |
| ccs | Nissan Leaf | SMALL | BEV | 4.16 | 28 | 916 | 34.2 | full 20-ch model | 0.54, 0.43, 0.03 | SMALL | OK | 17/28 | 59.1 |
| ccs | Nissan Rogue | SMALL | ICE | 4.31 | 5 | 100 | 63.5 | full 20-ch model | 0.67, 0.28, 0.05 | SMALL | OK | 3/5 | 84.6 |
| ccs | RAM 1500 | LARGE | ICE | 6.73 | 99 | 384 | 36.6 | full 20-ch model | 0.40, 0.14, 0.47 | LARGE | OK | 57/99 | 58.8 |
| ccs | RAM 2500/3500 HD | LARGE | ICE | 6.57 | 5 | 74 | 19.8 | full 20-ch model | 0.40, 0.11, 0.49 | LARGE | OK | 4/5 | 63.2 |
| ccs | Rivian R1T/R1S | LARGE | BEV | 5.58 | 96 | 386 | 44.9 | full 20-ch model | 0.37, 0.30, 0.33 | SMALL | MISS | 39/96 | 40.6 |
| ccs | SEAT Ateca | SMALL | ICE | 4.11 | 4 | 152 | 50.8 | full 20-ch model | 0.75, 0.23, 0.02 | SMALL | OK | 4/4 | 96.2 |
| ccs | Skoda Kamiq | SMALL | ICE | 4.04 | 1 | 35 | 88.1 | full 20-ch model | 0.86, 0.11, 0.03 | SMALL | OK | 1/1 | 100.0 |
| ccs | Skoda Karoq | SMALL | ICE | 4.15 | 3 | 8 | 16.3 | full 20-ch model | 0.86, 0.11, 0.03 | SMALL | OK | 3/3 | 100.0 |
| ccs | Skoda Kodiaq | SMALL (border) | ICE | 4.41 | 12 | 296 | 32.0 | full 20-ch model | 0.72, 0.24, 0.04 | SMALL | OK | 12/12 | 91.0 |
| ccs | Skoda Octavia | SMALL | ICE | 4.15 | 7 | 286 | 39.2 | full 20-ch model | 0.77, 0.21, 0.02 | SMALL | OK | 7/7 | 91.5 |
| ccs | Skoda Superb | MID | ICE | 4.48 | 10 | 315 | 42.6 | full 20-ch model | 0.66, 0.30, 0.04 | SMALL | MISS | 0/10 | 12.3 |
| ccs | Subaru Ascent | MID | ICE | 4.72 | 34 | 389 | 40.6 | full 20-ch model | 0.73, 0.21, 0.06 | SMALL | MISS | 0/34 | 5.7 |
| ccs | Subaru Forester | SMALL | ICE | 4.19 | 36 | 437 | 42.0 | full 20-ch model | 0.69, 0.26, 0.06 | SMALL | OK | 35/36 | 84.3 |
| ccs | Subaru Impreza | SMALL | ICE | 4.13 | 49 | 709 | 39.3 | full 20-ch model | 0.71, 0.24, 0.05 | SMALL | OK | 48/49 | 87.0 |
| ccs | Subaru Legacy | SMALL | ICE | 4.24 | 2 | 51 | 34.0 | full 20-ch model | 0.75, 0.20, 0.06 | SMALL | OK | 2/2 | 91.3 |
| ccs | Subaru Outback | SMALL | ICE | 4.23 | 46 | 491 | 40.5 | full 20-ch model | 0.64, 0.26, 0.10 | SMALL | OK | 40/46 | 77.4 |
| ccs | Tesla Model 3 | MID | BEV | 4.57 | 28 | 352 | 52.9 | full 20-ch model | 0.42, 0.53, 0.05 | MID | OK | 17/28 | 60.9 |
| ccs | Tesla Model Y | MID | BEV | 4.73 | 21 | 538 | 42.5 | full 20-ch model | 0.34, 0.59, 0.06 | MID | OK | 17/21 | 66.8 |
| ccs | Toyota Alphard | MID | unknown | 4.79 | 6 | 165 | 34.3 | full 20-ch model | 0.61, 0.33, 0.06 | SMALL | MISS | 0/6 | 17.1 |
| ccs | Toyota Avalon | MID | HEV | 4.53 | 37 | 1250 | 32.9 | full 20-ch model | 0.67, 0.29, 0.04 | SMALL | MISS | 1/37 | 14.8 |
| ccs | Toyota C-HR | SMALL | unknown | 4.08 | 9 | 173 | 53.8 | full 20-ch model | 0.74, 0.23, 0.02 | SMALL | OK | 9/9 | 88.9 |
| ccs | Toyota Camry | MID (border) | HEV | 4.47 | 167 | 781 | 43.0 | full 20-ch model | 0.67, 0.29, 0.04 | SMALL | MISS | 12/167 | 11.6 |
| ccs | Toyota Corolla | SMALL | HEV | 4.09 | 272 | 1952 | 43.4 | full 20-ch model | 0.65, 0.28, 0.06 | SMALL | OK | 259/272 | 83.4 |
| ccs | Toyota Highlander | MID | ICE | 4.60 | 225 | 765 | 38.4 | full 20-ch model | 0.64, 0.32, 0.05 | SMALL | MISS | 42/225 | 18.3 |
| ccs | Toyota Mirai | SMALL | FCEV | 4.28 | 6 | 296 | 27.6 | full 20-ch model | 0.66, 0.30, 0.04 | SMALL | OK | 6/6 | 84.3 |
| ccs | Toyota Prius | SMALL | HEV | 4.13 | 208 | 767 | 49.0 | full 20-ch model | 0.72, 0.26, 0.03 | SMALL | OK | 198/208 | 93.7 |
| ccs | Toyota Prius v | SMALL | HEV | 4.29 | 3 | 50 | 56.1 | full 20-ch model | 0.63, 0.32, 0.05 | SMALL | OK | 3/3 | 76.5 |
| ccs | Toyota RAV4 | SMALL | ICE | 4.34 | 450 | 2322 | 42.9 | full 20-ch model | 0.65, 0.31, 0.04 | SMALL | OK | 381/450 | 82.3 |
| ccs | Toyota Sienna | LARGE | ICE | 5.21 | 94 | 1854 | 35.3 | full 20-ch model | 0.50, 0.31, 0.19 | SMALL | MISS | 15/94 | 21.0 |
| ccs | Volkswagen Arteon | MID | unknown | 4.49 | 4 | 134 | 52.3 | full 20-ch model | 0.76, 0.22, 0.03 | SMALL | MISS | 0/4 | 7.0 |
| ccs | Volkswagen Atlas | LARGE | ICE | 5.11 | 32 | 961 | 38.0 | full 20-ch model | 0.55, 0.37, 0.09 | SMALL | MISS | 0/32 | 5.6 |
| ccs | Volkswagen Crafter | LARGE | unknown | 6.24 | 4 | 115 | 42.4 | full 20-ch model | 0.49, 0.23, 0.28 | SMALL | MISS | 0/4 | 42.9 |
| ccs | Volkswagen Golf | SMALL | unknown | 4.02 | 40 | 391 | 46.6 | full 20-ch model | 0.73, 0.24, 0.03 | SMALL | OK | 38/40 | 92.7 |
| ccs | Volkswagen Jetta | SMALL | ICE | 4.17 | 12 | 342 | 27.0 | full 20-ch model | 0.72, 0.24, 0.04 | SMALL | OK | 12/12 | 89.6 |
| ccs | Volkswagen Passat | SMALL (border) | unknown | 4.40 | 8 | 256 | 46.2 | full 20-ch model | 0.68, 0.28, 0.04 | SMALL | OK | 8/8 | 84.0 |
| ccs | Volkswagen T-Roc | SMALL | ICE | 3.99 | 3 | 147 | 24.7 | full 20-ch model | 0.79, 0.16, 0.05 | SMALL | OK | 3/3 | 92.7 |
| ccs | Volkswagen Taos | SMALL | ICE | 4.18 | 2 | 55 | 30.4 | full 20-ch model | 0.75, 0.23, 0.02 | SMALL | OK | 2/2 | 100.0 |
| ccs | Volkswagen Tiguan | SMALL | unknown | 4.33 | 31 | 385 | 38.2 | full 20-ch model | 0.71, 0.24, 0.04 | SMALL | OK | 30/31 | 91.2 |
| ccs | Volkswagen Touran | SMALL (border) | ICE | 4.39 | 1 | 44 | 9.2 | full 20-ch model | 0.70, 0.25, 0.05 | SMALL | OK | 1/1 | 96.4 |
| ccs | Volkswagen Transporter | LARGE (border) | ICE | 4.91 | 2 | 74 | 30.4 | full 20-ch model | 0.57, 0.36, 0.07 | SMALL | MISS | 0/2 | 2.0 |
| ccs_honda | Honda CR-V | SMALL | ICE | 4.28 | 96 | 486 | 45.0 | full 20-ch model | 0.76, 0.20, 0.04 | SMALL | OK | 95/96 | 93.5 |
| ccs_honda | Honda Civic | SMALL | ICE | 4.20 | 123 | 1173 | 43.7 | full 20-ch model | 0.70, 0.26, 0.04 | SMALL | OK | 121/123 | 88.7 |
| ccs_honda | Honda Fit | SMALL | ICE | 3.74 | 7 | 273 | 35.0 | full 20-ch model | 0.60, 0.36, 0.04 | SMALL | OK | 6/7 | 73.3 |
| ccs_honda | Honda Freed | SMALL | unknown | 4.06 | 1 | 48 | 25.6 | full 20-ch model | 0.55, 0.34, 0.10 | SMALL | OK | 1/1 | 80.0 |
| ccs_honda | Honda HR-V | SMALL | ICE | 4.01 | 5 | 152 | 33.3 | full 20-ch model | 0.73, 0.24, 0.03 | SMALL | OK | 5/5 | 96.2 |
| comma2k19 | Honda Civic | SMALL | ICE | 4.20 | 1 | 196 | 61.7 | full 20-ch model | 0.48, 0.49, 0.02 | MID | MISS | 0/1 | 52.4 |
| comma2k19 | Toyota RAV4 | SMALL | ICE | 4.13 | 1 | 185 | 55.7 | full 20-ch model | 0.54, 0.44, 0.02 | SMALL | OK | 1/1 | 57.1 |

### Pooled vehicle-level decisions over the external sets (primary run of each set; one physical car, archive or device x car model = one decision, judged by its own label)

bev_ice:

| scope | BEV cars correct | ICE cars correct | all cars |
|---|---|---|---|
| phase-1 sets (2.1-2.9) | 23/27 (85.2) | 268/284 (94.4) | 291/311 (93.6) |
| phase-1 without VED (264 ICE cars would dominate) | 23/24 (95.8) | 10/20 (50.0) | 33/44 (75.0) |
| longitudinal / speed-only sets only (V2 / V3 variants) | 20/24 (83.3) | 268/283 (94.7) | 288/307 (93.8) |
| commaSteeringControl archives (primary run per task; one archive = one pseudo-vehicle) | 8/9 (88.9) | 27/27 (100.0) | 35/36 (97.2) |
| MoreTorque cars (full model) | 0/1 (0.0) | 11/11 (100.0) | 11/12 (91.7) |
| commaCarSegments devices, ccs + ccs_honda (full model; one device x car model = one decision, judged by the device's own platform label; a device under more than one car model counts once per model) | 544/630 (86.3) | 1916/2148 (89.2) | 2460/2778 (88.6) |
| comma2k19 cars on 2018 hardware (full model; one device = one decision) | 0/0 (-) | 2/2 (100.0) | 2/2 (100.0) |
| all sets | 575/667 (86.2) | 2224/2472 (90.0) | 2799/3139 (89.2) |

size:

| scope | SMALL cars correct | MID cars correct | LARGE cars correct | all cars |
|---|---|---|---|---|
| phase-1 sets (2.1-2.9) | 25/27 (92.6) | 2/20 (10.0) | 0/13 (0.0) | 27/60 (45.0) |
| phase-1 without VED (264 ICE cars would dominate) | 22/24 (91.7) | 2/20 (10.0) | 0/13 (0.0) | 24/57 (42.1) |
| longitudinal / speed-only sets only (V2 / V3 variants) | 24/25 (96.0) | 1/19 (5.3) | 0/11 (0.0) | 25/55 (45.5) |
| commaSteeringControl archives (primary run per task; one archive = one pseudo-vehicle) | 15/17 (88.2) | 4/13 (30.8) | 4/6 (66.7) | 23/36 (63.9) |
| MoreTorque cars (full model) | 8/8 (100.0) | 0/4 (0.0) | 0/0 (-) | 8/12 (66.7) |
| commaCarSegments devices, ccs + ccs_honda (full model; one device x car model = one decision, judged by the device's own platform label; a device under more than one car model counts once per model) | 1863/2030 (91.8) | 360/1301 (27.7) | 250/1053 (23.7) | 2473/4384 (56.4) |
| comma2k19 cars on 2018 hardware (full model; one device = one decision) | 1/2 (50.0) | 0/0 (-) | 0/0 (-) | 1/2 (50.0) |
| all sets | 1912/2084 (91.7) | 366/1338 (27.4) | 254/1072 (23.7) | 2532/4494 (56.3) |

### Automation-engaged vs human-driven windows (primary powertrain run; window accuracy and P(BEV) per sub-population)

| dataset | windows | n | BEV | ICE | accuracy % | BEV recall % | ICE recall % | mean P(BEV) on BEV | mean P(BEV) on ICE |
|---|---|---|---|---|---|---|---|---|---|
| ultraav | DAS engaged | 1942 | 1285 | 657 | 71.6 | 90.4 | 34.9 | 0.79 | 0.60 |
| ultraav | human-driven | 45 | 45 | 0 | 95.6 | 95.6 | - | 0.87 | - |
| openacc | DAS engaged | 3019 | 1662 | 1357 | 68.4 | 86.3 | 46.4 | 0.77 | 0.53 |
| openacc | human-driven | 936 | 218 | 718 | 67.6 | 86.2 | 62.0 | 0.76 | 0.41 |
| uga_ev | DAS engaged | 164 | 164 | 0 | 44.5 | 44.5 | - | 0.47 | - |
| uga_ev | human-driven | 162 | 80 | 82 | 40.7 | 53.8 | 28.0 | 0.52 | 0.58 |
| microsim_ev | DAS engaged | 111 | 111 | 0 | 55.0 | 55.0 | - | 0.57 | - |
| openacc_500x | human-driven | 13523 | 0 | 13523 | 87.9 | - | 87.9 | - | 0.18 |
| tesla_yuan | human-driven | 2619 | 2619 | 0 | 62.3 | 62.3 | - | 0.61 | - |
| ved | human-driven | 39259 | 431 | 38828 | 68.8 | 17.4 | 69.3 | 0.25 | 0.34 |
| ccs_sample | DAS engaged | 2 | 1 | 1 | 50.0 | 100.0 | 0.0 | 0.87 | 0.83 |
| ccs_sample | human-driven | 2 | 2 | 0 | 100.0 | 100.0 | - | 0.99 | - |
| csc | DAS engaged | 51462 | 11963 | 39499 | 75.4 | 63.9 | 78.9 | 0.59 | 0.28 |
| moretorque | DAS engaged | 3961 | 251 | 3710 | 67.3 | 6.0 | 71.5 | 0.14 | 0.35 |
| moretorque | human-driven | 3229 | 623 | 2606 | 78.7 | 8.8 | 95.4 | 0.22 | 0.11 |
| ccs | DAS engaged | 11229 | 2359 | 8870 | 64.8 | 65.8 | 64.5 | 0.64 | 0.39 |
| ccs | human-driven | 18356 | 3061 | 15295 | 92.2 | 88.7 | 93.0 | 0.85 | 0.12 |
| ccs_honda | DAS engaged | 890 | 0 | 890 | 80.7 | - | 80.7 | - | 0.29 |
| ccs_honda | human-driven | 1194 | 0 | 1194 | 95.6 | - | 95.6 | - | 0.08 |
| comma2k19 | DAS engaged | 226 | 0 | 226 | 50.9 | - | 50.9 | - | 0.47 |
| comma2k19 | human-driven | 155 | 0 | 155 | 100.0 | - | 100.0 | - | 0.06 |

Same car, both automation states (cars with >= 5 windows in each state): mean P(BEV)

| dataset | car model | label | DAS windows | P(BEV) DAS | human windows | P(BEV) human |
|---|---|---|---|---|---|---|
| openacc | Audi A6 | ICE | 174 | 0.71 | 46 | 0.76 |
| openacc | Audi e-tron | BEV | 316 | 0.78 | 34 | 0.86 |
| openacc | BMW X5 | ICE | 174 | 0.61 | 46 | 0.73 |
| openacc | Ford S-Max | ICE | 183 | 0.42 | 34 | 0.47 |
| openacc | Jaguar I-Pace | BEV | 259 | 0.81 | 33 | 0.67 |
| openacc | Mazda 3 | ICE | 271 | 0.55 | 34 | 0.47 |
| openacc | Mercedes-Benz A-Class | ICE | 174 | 0.49 | 46 | 0.71 |
| openacc | Tesla Model 3 | BEV | 410 | 0.78 | 76 | 0.85 |
| openacc | Tesla Model S | BEV | 300 | 0.79 | 34 | 0.77 |
| openacc | Tesla Model X | BEV | 312 | 0.74 | 33 | 0.62 |
| openacc | Volkswagen e-Golf | BEV | 65 | 0.51 | 8 | 0.47 |
| openacc | Volvo XC40 | ICE | 161 | 0.77 | 23 | 0.47 |
| ultraav | Audi e-tron | BEV | 209 | 0.87 | 10 | 0.92 |
| ultraav | Jaguar I-Pace | BEV | 187 | 0.85 | 8 | 0.81 |
| ultraav | Tesla Model 3 | BEV | 376 | 0.77 | 12 | 0.94 |
| ultraav | Tesla Model S | BEV | 244 | 0.79 | 11 | 0.83 |
| uga_ev | Ford Mustang Mach-E | BEV | 56 | 0.32 | 27 | 0.44 |
| uga_ev | Kia Niro EV | BEV | 108 | 0.55 | 53 | 0.56 |

### OpenACC by campaign (primary runs: bev_ice V2 variant, size V3 variant; model-level = mean probability over the model's windows of that campaign)

| campaign | car model | win | DAS % | powertrain | P(BEV) | decision | size label (footprint) | P(S, M, L) | size decision |
|---|---|---|---|---|---|---|---|---|---|
| AstaZero | Audi A6 | 220 | 79.1 | ICE | 0.72 | BEV MISS | MID (4.75) | 0.65, 0.27, 0.08 | SMALL MISS |
| AstaZero | Audi A8 | 220 | 100.0 | ICE | 0.22 | ICE OK | LARGE (4.91) | 0.64, 0.25, 0.12 | SMALL MISS |
| AstaZero | BMW X5 | 220 | 79.1 | ICE | 0.64 | BEV MISS | LARGE (4.99) | 0.66, 0.27, 0.07 | SMALL MISS |
| AstaZero | Mercedes-Benz A-Class | 220 | 79.1 | ICE | 0.53 | BEV MISS | SMALL (4.25) | 0.65, 0.27, 0.07 | SMALL OK |
| AstaZero | Tesla Model 3 | 220 | 79.1 | BEV | 0.80 | BEV OK | MID (4.57) | 0.64, 0.28, 0.08 | SMALL MISS |
| Casale | SsangYong Rexton | 69 | 0.0 | ICE | 0.38 | ICE OK | MID (4.70) | 0.62, 0.31, 0.07 | SMALL MISS |
| Cherasco | Fiat 500X | 182 | 0.0 | ICE | 0.30 | ICE OK | SMALL (3.97) | 0.64, 0.16, 0.20 | SMALL OK |
| Cherasco | Volkswagen Polo | 82 | 0.0 | ICE | 0.30 | ICE OK | SMALL (3.61) | 0.25, 0.30, 0.45 | LARGE MISS |
| Cherasco | Volvo XC40 | 184 | 86.5 | ICE | 0.73 | BEV MISS | SMALL (4.36) | 0.72, 0.15, 0.14 | SMALL OK |
| JRC low speed | Ford S-Max | 103 | 90.8 | ICE | 0.34 | ICE OK | MID (4.57) | 0.49, 0.31, 0.20 | SMALL MISS |
| Vicolungo | Ford S-Max | 114 | 76.1 | ICE | 0.51 | BEV MISS | MID (4.57) | 0.57, 0.23, 0.19 | SMALL MISS |
| Vicolungo | MINI Cooper | 42 | 0.0 | ICE | 0.39 | ICE OK | SMALL (3.74) | 0.52, 0.23, 0.25 | SMALL OK |
| Vicolungo | Mitsubishi Space Star | 114 | 0.0 | ICE | 0.29 | ICE OK | SMALL (3.48) | 0.64, 0.17, 0.20 | SMALL OK |
| Vicolungo | Volkswagen e-Golf | 73 | 90.1 | BEV | 0.51 | BEV OK | SMALL (4.02) | 0.71, 0.17, 0.11 | SMALL OK |
| ZalaZone | Audi A4 Avant | 225 | 92.4 | HEV | 0.71 | BEV (shares only) | SMALL (4.41) | 0.56, 0.36, 0.08 | SMALL OK |
| ZalaZone | Audi e-tron | 350 | 90.3 | BEV | 0.79 | BEV OK | MID (4.84) | 0.51, 0.36, 0.13 | SMALL MISS |
| ZalaZone | Jaguar I-Pace | 292 | 88.7 | BEV | 0.79 | BEV OK | LARGE (4.92) | 0.57, 0.33, 0.09 | SMALL MISS |
| ZalaZone | Mazda 3 | 305 | 88.9 | ICE | 0.54 | BEV MISS | SMALL (4.29) | 0.51, 0.34, 0.15 | SMALL OK |
| ZalaZone | Mercedes-Benz GLE 450 4MATIC | 360 | 91.1 | HEV | 0.49 | ICE (shares only) | LARGE (5.02) | 0.59, 0.33, 0.08 | SMALL MISS |
| ZalaZone | Tesla Model 3 | 266 | 88.7 | BEV | 0.78 | BEV OK | MID (4.57) | 0.54, 0.36, 0.10 | SMALL MISS |
| ZalaZone | Tesla Model S | 334 | 89.8 | BEV | 0.79 | BEV OK | LARGE (4.97) | 0.55, 0.35, 0.10 | SMALL MISS |
| ZalaZone | Tesla Model X | 345 | 90.4 | BEV | 0.73 | BEV OK | LARGE (5.06) | 0.53, 0.35, 0.12 | SMALL MISS |

### VED: size shares per weight bin (V3 size variant; the only ordinal check possible without footprints) and powertrain decisions per vehicle

Pooled over powertrains; `ordinal score` = 1 x P(SMALL) + 2 x P(MID) + 3 x P(LARGE) of the mean probability (rises with size if the model reads size):

| weight bin | vehicles | windows | predicted share SMALL, MID, LARGE % | mean P(SMALL, MID, LARGE) | ordinal score |
|---|---|---|---|---|---|
| 2000 lb | 1 | 164 | 48.2, 36.6, 15.2 | 0.396, 0.367, 0.237 | 1.841 |
| 2500 lb | 42 | 6406 | 45.4, 46.0, 8.6 | 0.413, 0.396, 0.191 | 1.779 |
| 3000 lb | 107 | 17342 | 46.9, 42.3, 10.8 | 0.410, 0.384, 0.207 | 1.797 |
| 3500 lb | 114 | 18507 | 46.2, 45.6, 8.2 | 0.416, 0.392, 0.191 | 1.775 |
| 4000 lb | 53 | 8242 | 52.1, 41.4, 6.5 | 0.442, 0.387, 0.172 | 1.730 |
| 4500 lb | 37 | 5011 | 40.9, 48.3, 10.8 | 0.394, 0.398, 0.208 | 1.814 |
| 5000 lb | 6 | 777 | 49.2, 45.9, 4.9 | 0.445, 0.384, 0.171 | 1.726 |
| 5500 lb | 4 | 331 | 43.5, 52.0, 4.5 | 0.421, 0.410, 0.169 | 1.749 |
| 6000 lb | 3 | 198 | 45.5, 47.0, 7.6 | 0.432, 0.387, 0.180 | 1.748 |
| 6500 lb | 1 | 15 | 60.0, 33.3, 6.7 | 0.443, 0.388, 0.168 | 1.725 |
| n/a | 16 | 2570 | 40.4, 52.3, 7.4 | 0.398, 0.415, 0.187 | 1.789 |

Vehicle-level Spearman correlation between the weight bin and the ordinal score: -0.125 (n = 368 vehicles with a weight bin).

Powertrain (V3 variant): share of windows / drives / vehicles called BEV, per true powertrain (BEV and ICE are scored; HEV and PHEV are shares only):

| true powertrain | vehicles | drives | windows | windows called BEV % | drives called BEV % | vehicles called BEV % | mean P(BEV) | median vehicle P(BEV) |
|---|---|---|---|---|---|---|---|---|
| BEV | 3 | 317 | 431 | 17.4 | 16.4 | 0.0 | 0.25 | 0.22 |
| HEV | 93 | 7456 | 16206 | 24.3 | 18.2 | 1.1 | 0.29 | 0.29 |
| ICE | 264 | 15453 | 38828 | 30.7 | 24.2 | 2.3 | 0.34 | 0.34 |
| PHEV | 24 | 2354 | 4098 | 22.6 | 18.1 | 0.0 | 0.28 | 0.27 |

### commaSteeringControl (`csc`): decisions per archive (= pseudo-vehicle = 2023 openpilot platform), all runs

Mean probability over the archive's windows -> arg-max; `OK` / `MISS` against the archive label; stratum = the car model's status in the lab's `split_platform` split (seen = training / validation model, held-out = one of the 41 test models). Size runs: V1 variant (primary), full model under the V1 mask (secondary), V3 speed-only variant; powertrain runs: V3 speed-only (primary, RESULTS.md 1.3), V1 (the matched input list), full + V1 mask.

| archive | model | footprint | size | pt | stratum | win | DAS % | size V1: P(S, M, L) % -> decision | size full+V1 mask | size V3 | BEV vs ICE V3: P(BEV) % -> decision | V1: P(BEV) % -> decision | full+V1 mask: P(BEV) % -> decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CHEVROLET_BOLT_EUV_2022 | Chevrolet Bolt EUV | 3.99 | SMALL | BEV | seen | 1500 | 90.7 | 54, 30, 16 -> SMALL OK | SMALL OK | SMALL OK | 66 -> BEV OK | 35 -> ICE MISS | 31 -> ICE MISS |
| CHEVROLET_SILVERADO_1500_2020 | Chevrolet Silverado 1500 | 6.65 | LARGE | ICE | seen | 1500 | 92.4 | 28, 34, 38 -> LARGE OK | SMALL MISS | LARGE OK | 34 -> ICE OK | 14 -> ICE OK | 20 -> ICE OK |
| GENESIS_GV60_ELECTRIC_1ST_GEN | Genesis GV60 | 4.73 | MID | BEV | held-out | 999 | 94.9 | 38, 43, 19 -> MID OK | SMALL MISS | SMALL MISS | 58 -> BEV OK | 19 -> ICE MISS | 29 -> ICE MISS |
| HONDA_ACCORD_2018 | Honda Accord | 4.54 | MID | ICE | seen | 1500 | 91.9 | 52, 35, 13 -> SMALL MISS | SMALL MISS | SMALL MISS | 24 -> ICE OK | 9 -> ICE OK | 18 -> ICE OK |
| HONDA_CIVIC_2016 | Honda Civic | 4.20 | SMALL | ICE | seen | 1500 | 91.9 | 59, 35, 6 -> SMALL OK | SMALL OK | SMALL OK | 28 -> ICE OK | 18 -> ICE OK | 18 -> ICE OK |
| HONDA_CIVIC_2022 | Honda Civic | 4.21 | SMALL | ICE | seen | 1500 | 91.6 | 58, 37, 5 -> SMALL OK | SMALL OK | SMALL OK | 29 -> ICE OK | 17 -> ICE OK | 21 -> ICE OK |
| HONDA_CIVIC_BOSCH_2019 | Honda Civic | 4.20 | SMALL | ICE | seen | 1500 | 90.4 | 57, 36, 6 -> SMALL OK | SMALL OK | SMALL OK | 27 -> ICE OK | 20 -> ICE OK | 18 -> ICE OK |
| HONDA_CR_V_2016 | Honda CR-V | 4.23 | SMALL | ICE | seen | 1000 | 88.7 | 40, 13, 48 -> LARGE MISS | SMALL OK | SMALL OK | 27 -> ICE OK | 12 -> ICE OK | 17 -> ICE OK |
| HONDA_CR_V_2017 | Honda CR-V | 4.28 | SMALL | ICE | seen | 1500 | 92.2 | 54, 29, 17 -> SMALL OK | SMALL OK | SMALL OK | 25 -> ICE OK | 11 -> ICE OK | 19 -> ICE OK |
| HONDA_ODYSSEY_2018 | Honda Odyssey | 5.14 | LARGE | ICE | seen | 1500 | 91.6 | 43, 34, 23 -> SMALL MISS | SMALL MISS | SMALL MISS | 37 -> ICE OK | 9 -> ICE OK | 20 -> ICE OK |
| HONDA_PILOT_2017 | Honda Pilot | 4.91 (border) | LARGE | ICE | seen | 1500 | 92.9 | 28, 20, 52 -> LARGE OK | SMALL MISS | LARGE OK | 22 -> ICE OK | 8 -> ICE OK | 16 -> ICE OK |
| HYUNDAI_ELANTRA_2021 | Hyundai Elantra | 4.31 | SMALL | ICE | seen | 1500 | 90.6 | 56, 36, 7 -> SMALL OK | SMALL OK | SMALL OK | 31 -> ICE OK | 13 -> ICE OK | 21 -> ICE OK |
| HYUNDAI_IONIQ_5_2022 | Hyundai Ioniq 5 | 4.83 (border) | MID | BEV | seen | 1500 | 92.3 | 37, 43, 20 -> MID OK | SMALL MISS | SMALL MISS | 66 -> BEV OK | 24 -> ICE MISS | 32 -> ICE MISS |
| HYUNDAI_IONIQ_ELECTRIC_LIMITED_2019 | Hyundai Ioniq | 4.22 | SMALL | BEV | held-out | 497 | 92.8 | 40, 51, 9 -> MID MISS | SMALL OK | SMALL OK | 32 -> ICE MISS | 25 -> ICE MISS | 21 -> ICE MISS |
| HYUNDAI_KONA_ELECTRIC_2019 | Hyundai Kona | 4.02 | SMALL | BEV | held-out | 1500 | 92.6 | 56, 35, 10 -> SMALL OK | SMALL OK | SMALL OK | 55 -> BEV OK | 35 -> ICE MISS | 32 -> ICE MISS |
| HYUNDAI_PALISADE_2020 | Hyundai Palisade | 4.96 | LARGE | ICE | seen | 1500 | 93.3 | 44, 36, 19 -> SMALL MISS | SMALL MISS | SMALL MISS | 33 -> ICE OK | 9 -> ICE OK | 20 -> ICE OK |
| HYUNDAI_SANTA_FE_2019 | Hyundai Santa Fe | 4.53 | MID | ICE | held-out | 1500 | 92.5 | 51, 28, 20 -> SMALL MISS | SMALL MISS | SMALL MISS | 31 -> ICE OK | 12 -> ICE OK | 20 -> ICE OK |
| HYUNDAI_SANTA_FE_2022 | Hyundai Santa Fe | 4.57 | MID | ICE | held-out | 1500 | 93.3 | 50, 29, 21 -> SMALL MISS | SMALL MISS | SMALL MISS | 28 -> ICE OK | 10 -> ICE OK | 21 -> ICE OK |
| HYUNDAI_SONATA_2019 | Hyundai Sonata | 4.56 | MID | ICE | seen | 1500 | 89.8 | 58, 35, 7 -> SMALL MISS | SMALL MISS | SMALL MISS | 32 -> ICE OK | 22 -> ICE OK | 19 -> ICE OK |
| HYUNDAI_SONATA_2020 | Hyundai Sonata | 4.62 | MID | ICE | seen | 1500 | 91.6 | 48, 29, 22 -> SMALL MISS | SMALL MISS | SMALL MISS | 32 -> ICE OK | 11 -> ICE OK | 20 -> ICE OK |
| HYUNDAI_TUCSON_4TH_GEN | Hyundai Tucson | 4.53 | MID | ICE | seen | 999 | 93.2 | 39, 45, 16 -> MID OK | SMALL MISS | SMALL MISS | 25 -> ICE OK | 8 -> ICE OK | 18 -> ICE OK |
| KIA_EV6_2022 | Kia EV6 | 4.74 | MID | BEV | seen | 1500 | 93.1 | 55, 36, 10 -> SMALL MISS | SMALL MISS | SMALL MISS | 65 -> BEV OK | 49 -> ICE MISS | 32 -> ICE MISS |
| KIA_NIRO_EV_2020 | Kia Niro | 4.27 | SMALL | BEV | seen | 1500 | 90.4 | 56, 35, 9 -> SMALL OK | SMALL OK | SMALL OK | 60 -> BEV OK | 38 -> ICE MISS | 31 -> ICE MISS |
| NISSAN_LEAF_2018 | Nissan Leaf | 4.16 | SMALL | BEV | held-out | 1500 | 90.1 | 49, 32, 20 -> SMALL OK | SMALL OK | SMALL OK | 57 -> BEV OK | 28 -> ICE MISS | 35 -> ICE MISS |
| NISSAN_LEAF_2018_Instrument_Cluster | Nissan Leaf | 4.16 | SMALL | BEV | held-out | 1467 | 91.9 | 57, 31, 13 -> SMALL OK | SMALL OK | SMALL OK | 53 -> BEV OK | 18 -> ICE MISS | 30 -> ICE MISS |
| RAM_1500_5TH_GEN | RAM 1500 | 6.73 | LARGE | ICE | seen | 1500 | 94.1 | 36, 25, 39 -> LARGE OK | SMALL MISS | LARGE OK | 33 -> ICE OK | 13 -> ICE OK | 17 -> ICE OK |
| TOYOTA_CAMRY_2018 | Toyota Camry | 4.47 (border) | MID | ICE | seen | 1500 | 91.7 | 57, 34, 9 -> SMALL MISS | SMALL MISS | SMALL MISS | 22 -> ICE OK | 11 -> ICE OK | 24 -> ICE OK |
| TOYOTA_CAMRY_2021 | Toyota Camry | 4.47 (border) | MID | ICE | seen | 1500 | 92.7 | 58, 35, 7 -> SMALL MISS | SMALL MISS | SMALL MISS | 16 -> ICE OK | 11 -> ICE OK | 20 -> ICE OK |
| TOYOTA_COROLLA_2017 | Toyota Corolla | 4.14 | SMALL | ICE | held-out | 1500 | 90.6 | 44, 30, 26 -> SMALL OK | SMALL OK | SMALL OK | 37 -> ICE OK | 8 -> ICE OK | 17 -> ICE OK |
| TOYOTA_COROLLA_TSS2_2019 | Toyota Corolla | 4.09 | SMALL | ICE | held-out | 1500 | 92.1 | 56, 35, 10 -> SMALL OK | SMALL OK | SMALL OK | 20 -> ICE OK | 8 -> ICE OK | 19 -> ICE OK |
| TOYOTA_HIGHLANDER_2017 | Toyota Highlander | 4.60 | MID | ICE | seen | 1500 | 93.5 | 31, 35, 34 -> MID OK | SMALL MISS | SMALL MISS | 30 -> ICE OK | 4 -> ICE OK | 16 -> ICE OK |
| TOYOTA_HIGHLANDER_2020 | Toyota Highlander | 4.68 | MID | ICE | seen | 1500 | 92.6 | 52, 37, 11 -> SMALL MISS | SMALL MISS | SMALL MISS | 21 -> ICE OK | 9 -> ICE OK | 18 -> ICE OK |
| TOYOTA_RAV4_2017 | Toyota RAV4 | 4.13 | SMALL | ICE | seen | 1500 | 93.0 | 47, 31, 22 -> SMALL OK | SMALL OK | SMALL OK | 22 -> ICE OK | 9 -> ICE OK | 19 -> ICE OK |
| TOYOTA_RAV4_2019 | Toyota RAV4 | 4.34 | SMALL | ICE | seen | 1500 | 91.6 | 55, 33, 12 -> SMALL OK | SMALL OK | SMALL OK | 18 -> ICE OK | 9 -> ICE OK | 21 -> ICE OK |
| TOYOTA_RAV4_2022 | Toyota RAV4 | 4.34 | SMALL | ICE | seen | 1500 | 93.2 | 57, 31, 12 -> SMALL OK | SMALL OK | SMALL OK | 16 -> ICE OK | 7 -> ICE OK | 23 -> ICE OK |
| TOYOTA_SIENNA_2018 | Toyota Sienna | 5.21 | LARGE | ICE | held-out | 1500 | 93.1 | 34, 27, 39 -> LARGE OK | SMALL MISS | SMALL MISS | 40 -> ICE OK | 7 -> ICE OK | 16 -> ICE OK |

### commaSteeringControl: pooled per stratum (window = segment = drive here; archive = pseudo-vehicle; model = car model)

size (SMALL / MID / LARGE):

| run | stratum | windows / archives / models | window acc / F1 % | window recall per class % | archive acc / F1 % | archives right per class | model acc % | models right per class | chance uniform / majority % |
|---|---|---|---|---|---|---|---|---|---|
| V1 | all | 51462 / 36 / 25 | 48.4 / 44.1 | SMALL 68.1 / MID 29.3 / LARGE 35.1 | 63.9 / 62.3 | SMALL 15/17 / MID 4/13 / LARGE 4/6 | 64.0 | SMALL 9/10 / MID 3/9 / LARGE 4/6 | 33.3 / 46.6 |
|  | seen | 37999 / 26 / 18 | 48.1 / 44.7 | SMALL 71.4 / MID 29.8 / LARGE 34.0 | 61.5 / 59.8 | SMALL 10/11 / MID 3/10 / LARGE 3/5 | 61.1 | SMALL 6/6 / MID 2/7 / LARGE 3/5 | 33.3 / 42.1 |
|  | held-out | 13463 / 10 / 7 | 49.1 / 41.8 | SMALL 61.5 / MID 27.3 / LARGE 40.9 | 70.0 / 72.3 | SMALL 5/6 / MID 1/3 / LARGE 1/1 | 71.4 | SMALL 3/4 / MID 1/2 / LARGE 1/1 | 33.3 / 59.2 |
| full+V1 mask | all | 51462 / 36 / 25 | 47.6 / 29.7 | SMALL 94.8 / MID 4.9 / LARGE 9.6 | 47.2 / 21.4 | SMALL 17/17 / MID 0/13 / LARGE 0/6 | 40.0 | SMALL 10/10 / MID 0/9 / LARGE 0/6 | 33.3 / 46.6 |
|  | seen | 37999 / 26 / 18 | 43.8 / 28.3 | SMALL 94.9 / MID 5.4 / LARGE 9.0 | 42.3 / 19.8 | SMALL 11/11 / MID 0/10 / LARGE 0/5 | 33.3 | SMALL 6/6 / MID 0/7 / LARGE 0/5 | 33.3 / 42.1 |
|  | held-out | 13463 / 10 / 7 | 58.4 / 33.6 | SMALL 94.7 / MID 3.3 / LARGE 13.1 | 60.0 / 25.0 | SMALL 6/6 / MID 0/3 / LARGE 0/1 | 57.1 | SMALL 4/4 / MID 0/2 / LARGE 0/1 | 33.3 / 59.2 |
| V3 | all | 51462 / 36 / 25 | 47.0 / 42.5 | SMALL 65.3 / MID 30.1 / LARGE 32.8 | 55.6 / 44.9 | SMALL 17/17 / MID 0/13 / LARGE 3/6 | 52.0 | SMALL 10/10 / MID 0/9 / LARGE 3/6 | 33.3 / 46.6 |
|  | seen | 37999 / 26 / 18 | 45.2 / 42.9 | SMALL 62.9 / MID 30.1 / LARGE 36.5 | 53.8 / 46.6 | SMALL 11/11 / MID 0/10 / LARGE 3/5 | 50.0 | SMALL 6/6 / MID 0/7 / LARGE 3/5 | 33.3 / 42.1 |
|  | held-out | 13463 / 10 / 7 | 52.1 / 38.2 | SMALL 70.3 / MID 29.8 / LARGE 14.7 | 60.0 / 25.0 | SMALL 6/6 / MID 0/3 / LARGE 0/1 | 57.1 | SMALL 4/4 / MID 0/2 / LARGE 0/1 | 33.3 / 59.2 |

bev_ice (BEV / ICE):

| run | stratum | windows / archives / models | window acc / F1 % | window recall per class % | archive acc / F1 % | archives right per class | model acc % | models right per class | chance uniform / majority % |
|---|---|---|---|---|---|---|---|---|---|
| V3 | all | 51462 / 36 / 25 | 75.4 / 69.0 | BEV 63.9 / ICE 78.9 | 97.2 / 96.1 | BEV 8/9 / ICE 27/27 | 96.0 | BEV 7/8 / ICE 17/17 | 50.0 / 76.8 |
|  | seen | 37999 / 26 / 18 | 78.6 / 68.7 | BEV 70.9 / ICE 80.0 | 100.0 / 100.0 | BEV 4/4 / ICE 22/22 | 100.0 | BEV 4/4 / ICE 14/14 | 50.0 / 84.2 |
|  | held-out | 13463 / 10 / 7 | 66.6 / 65.7 | BEV 56.9 / ICE 74.3 | 90.0 / 89.9 | BEV 4/5 / ICE 5/5 | 85.7 | BEV 3/4 / ICE 3/3 | 50.0 / 55.7 |
| V1 | all | 51462 / 36 / 25 | 79.4 / 61.7 | BEV 24.4 / ICE 96.1 | 75.0 / 42.9 | BEV 0/9 / ICE 27/27 | 68.0 | BEV 0/8 / ICE 17/17 | 50.0 / 76.8 |
|  | seen | 37999 / 26 / 18 | 85.6 / 66.4 | BEV 31.6 / ICE 95.7 | 84.6 / 45.8 | BEV 0/4 / ICE 22/22 | 77.8 | BEV 0/4 / ICE 14/14 | 50.0 / 84.2 |
|  | held-out | 13463 / 10 / 7 | 62.1 / 51.4 | BEV 17.2 / ICE 97.8 | 50.0 / 33.3 | BEV 0/5 / ICE 5/5 | 42.9 | BEV 0/4 / ICE 3/3 | 50.0 / 55.7 |
| full+V1 mask | all | 51462 / 36 / 25 | 77.8 / 57.3 | BEV 18.3 / ICE 95.8 | 75.0 / 42.9 | BEV 0/9 / ICE 27/27 | 68.0 | BEV 0/8 / ICE 17/17 | 50.0 / 76.8 |
|  | seen | 37999 / 26 / 18 | 83.4 / 58.3 | BEV 18.4 / ICE 95.6 | 84.6 / 45.8 | BEV 0/4 / ICE 22/22 | 77.8 | BEV 0/4 / ICE 14/14 | 50.0 / 84.2 |
|  | held-out | 13463 / 10 / 7 | 61.8 / 51.8 | BEV 18.2 / ICE 96.5 | 50.0 / 33.3 | BEV 0/5 / ICE 5/5 | 42.9 | BEV 0/4 / ICE 3/3 | 50.0 / 55.7 |

Size, V1 variant: predicted class shares of the windows per true class and stratum (confusion as shares):

| stratum | true class | archives | windows | predicted share SMALL, MID, LARGE % | mean P(SMALL, MID, LARGE) |
|---|---|---|---|---|---|
| all | SMALL | 17 | 23964 | 68.1, 22.7, 9.2 | 0.53, 0.33, 0.14 |
| all | MID | 13 | 18498 | 60.1, 29.3, 10.7 | 0.49, 0.35, 0.16 |
| all | LARGE | 6 | 9000 | 39.6, 25.3, 35.1 | 0.35, 0.30, 0.35 |
| seen | SMALL | 11 | 16000 | 71.4, 20.0, 8.6 | 0.54, 0.32, 0.13 |
| seen | MID | 10 | 14499 | 60.8, 29.8, 9.4 | 0.49, 0.36, 0.15 |
| seen | LARGE | 5 | 7500 | 39.7, 26.3, 34.0 | 0.36, 0.30, 0.34 |
| held-out | SMALL | 6 | 7964 | 61.5, 28.1, 10.4 | 0.51, 0.34, 0.15 |
| held-out | MID | 3 | 3999 | 57.5, 27.3, 15.2 | 0.48, 0.32, 0.20 |
| held-out | LARGE | 1 | 1500 | 39.1, 20.0, 40.9 | 0.34, 0.27, 0.39 |

### MoreTorque (`moretorque`): decisions per vehicle, full 20-channel models

One row per physical car (pseudonymous ids); `lat-only %` = share of samples with the fork's lateral-only automation active (`latActive` without `enabled`, not counted in `das_enabled`); drive = run of consecutive segments.

| vehicle | model | size / pt | stratum | drives | win | DAS % | lat-only % | size: P(S, M, L) % -> decision | size drive acc % | BEV vs ICE: P(BEV) % -> decision | pt drive acc % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| veh_1c1bc25c | Hyundai Sonata | MID / ICE | seen | 297 | 2926 | 63.4 | 27.9 | 52, 23, 25 -> SMALL MISS | 13.8 | 31 -> ICE OK | 90.6 |
| veh_1e5bb0ca | Mazda 3 | SMALL / ICE | absent | 28 | 35 | 5.4 | 0.4 | 63, 27, 10 -> SMALL OK | 82.1 | 17 -> ICE OK | 92.9 |
| veh_5adb5f7d | Mazda 3 | SMALL / ICE | absent | 32 | 174 | 79.6 | 13.0 | 52, 24, 23 -> SMALL OK | 93.8 | 43 -> ICE OK | 75.0 |
| veh_660276e0 | Mazda 3 | SMALL / ICE | absent | 3 | 3 | 97.5 | 0.0 | 87, 5, 8 -> SMALL OK | 100.0 | 9 -> ICE OK | 100.0 |
| veh_8a7fb9d8 | Mazda 3 | SMALL / ICE | absent | 184 | 1211 | 71.2 | 23.4 | 78, 17, 5 -> SMALL OK | 95.1 | 26 -> ICE OK | 92.9 |
| veh_8b094748 | Mazda 3 | SMALL / ICE | absent | 8 | 11 | 0.0 | 100.0 | 63, 28, 9 -> SMALL OK | 62.5 | 24 -> ICE OK | 62.5 |
| veh_93d1a56f | Omoda E5 | SMALL / BEV | absent | 89 | 874 | 29.6 | 10.7 | 55, 24, 20 -> SMALL OK | 57.3 | 20 -> ICE MISS | 4.5 |
| veh_95a2afec | Mazda 3 | SMALL / ICE | absent | 150 | 878 | 77.4 | 0.3 | 74, 18, 8 -> SMALL OK | 91.3 | 18 -> ICE OK | 95.3 |
| veh_a1531e81 | Mazda 3 | SMALL / ICE | absent | 7 | 7 | 26.3 | 31.0 | 56, 29, 16 -> SMALL OK | 71.4 | 24 -> ICE OK | 85.7 |
| veh_d632de27 | Mazda CX-50 | MID / ICE | absent | 20 | 97 | 1.0 | 70.4 | 56, 26, 18 -> SMALL MISS | 10.0 | 7 -> ICE OK | 100.0 |
| veh_f1e45994 | Mazda CX-50 | MID / ICE | absent | 227 | 419 | 33.4 | 0.0 | 65, 16, 19 -> SMALL MISS | 5.3 | 17 -> ICE OK | 93.4 |
| veh_f20695f6 | Mazda CX-50 | MID / ICE | absent | 110 | 555 | 1.7 | 82.6 | 65, 26, 8 -> SMALL MISS | 8.2 | 7 -> ICE OK | 99.1 |

### MoreTorque: automation state per model (full models; window-level accuracy and mean probabilities)

`human` = das_frac < 0.5 and lat-only < 0.5; `lateral-only` = lat-only >= 0.5 (fork steering while `das_enabled` = 0); `engaged` = das_frac >= 0.5 (controller or stock cruise enabled).

| model | state | windows | size window acc % | mean P(S, M, L) | pt window acc % | mean P(BEV) |
|---|---|---|---|---|---|---|
| Hyundai Sonata | human | 234 | 29.1 | 0.54, 0.35, 0.10 | 97.0 | 0.08 |
| Hyundai Sonata | lateral-only | 839 | 25.1 | 0.44, 0.28, 0.27 | 98.0 | 0.07 |
| Hyundai Sonata | engaged | 1853 | 6.5 | 0.56, 0.19, 0.26 | 58.2 | 0.45 |
| Mazda 3 | human | 316 | 76.3 | 0.62, 0.29, 0.09 | 93.0 | 0.13 |
| Mazda 3 | lateral-only | 293 | 84.6 | 0.71, 0.23, 0.06 | 82.6 | 0.27 |
| Mazda 3 | engaged | 1710 | 90.4 | 0.77, 0.15, 0.07 | 84.8 | 0.26 |
| Mazda CX-50 | human | 356 | 12.1 | 0.61, 0.22, 0.17 | 95.8 | 0.11 |
| Mazda CX-50 | lateral-only | 568 | 16.2 | 0.65, 0.26, 0.09 | 98.8 | 0.07 |
| Mazda CX-50 | engaged | 147 | 0.0 | 0.71, 0.07, 0.23 | 83.0 | 0.27 |
| Omoda E5 | human | 614 | 63.7 | 0.55, 0.26, 0.20 | 8.8 | 0.22 |
| Omoda E5 | lateral-only | 9 | 88.9 | 0.71, 0.07, 0.22 | 11.1 | 0.33 |
| Omoda E5 | engaged | 251 | 62.5 | 0.56, 0.22, 0.22 | 6.0 | 0.14 |

### MoreTorque: speed regime per model (window mean speed; full models)

| model | mean speed | windows (>= 10) | size window acc % | mean P(S, M, L) | pt window acc % | mean P(BEV) |
|---|---|---|---|---|---|---|
| Hyundai Sonata | 0-10 m/s | 373 | 33.2 | 0.51, 0.37, 0.12 | 98.9 | 0.07 |
| Hyundai Sonata | 10-20 m/s | 466 | 24.7 | 0.45, 0.29, 0.26 | 98.5 | 0.06 |
| Hyundai Sonata | 20-30 m/s | 890 | 9.7 | 0.57, 0.20, 0.22 | 69.2 | 0.35 |
| Hyundai Sonata | 30- m/s | 1197 | 6.2 | 0.52, 0.18, 0.31 | 57.1 | 0.45 |
| Mazda 3 | 0-10 m/s | 357 | 72.8 | 0.60, 0.30, 0.10 | 89.9 | 0.17 |
| Mazda 3 | 10-20 m/s | 541 | 86.7 | 0.72, 0.21, 0.06 | 89.6 | 0.19 |
| Mazda 3 | 20-30 m/s | 767 | 94.3 | 0.83, 0.12, 0.05 | 81.1 | 0.29 |
| Mazda 3 | 30- m/s | 654 | 89.0 | 0.75, 0.16, 0.09 | 85.3 | 0.27 |
| Mazda CX-50 | 0-10 m/s | 371 | 19.7 | 0.61, 0.30, 0.10 | 97.8 | 0.08 |
| Mazda CX-50 | 10-20 m/s | 365 | 10.1 | 0.68, 0.21, 0.10 | 97.5 | 0.09 |
| Mazda CX-50 | 20-30 m/s | 272 | 6.6 | 0.66, 0.14, 0.21 | 91.2 | 0.17 |
| Mazda CX-50 | 30- m/s | 63 | 11.1 | 0.58, 0.18, 0.25 | 90.5 | 0.13 |
| Omoda E5 | 0-10 m/s | 546 | 54.2 | 0.49, 0.34, 0.18 | 3.3 | 0.17 |
| Omoda E5 | 10-20 m/s | 219 | 80.8 | 0.68, 0.08, 0.24 | 16.4 | 0.26 |
| Omoda E5 | 20-30 m/s | 109 | 76.1 | 0.61, 0.11, 0.27 | 14.7 | 0.22 |

### commaCarSegments (`ccs`): decisions per platform, full 20-channel models (V1 / V3 / V2 variants for comparison)

One row per openpilot platform; the devices are disjoint from the lab library (vehicle = device pseudonym, drive = device x route, one window per segment). Sorted by stratum (seen = training / validation model of the lab's `split_platform` split, held-out = one of the 41 test models, absent = not in the lab inventory), then footprint. `label` = source of footprint and model (lab table / public specs); `lead` = share of the windows per lead source (radar tracks / stock-ACC object / none = vision-only platform); `DAS %` = mean share of automation-engaged samples (driving automation system: stock cruise or controls allowed). Model-level decision = mean probability over the platform's labelled windows -> arg-max (`OK` / `MISS`); `veh right` = devices whose mean probability picks the label; HEV / PHEV / FCEV / MIXED platforms are unlabelled for the powertrain task and show the BEV probability and the predicted class only.

| platform | model | footprint | size | powertrain | stratum | label | lead % | veh | drives | win | DAS % | size full: P(S, M, L) % -> decision | size veh right | size drive acc % | size V1 | size V3 | BEV vs ICE full: P(BEV) % -> decision | pt veh right | pt V3 | pt V2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HONDA_E | Honda e | 3.80 | SMALL | BEV | seen | public | none 100 | 1 | 1 | 2 | 99.1 | 75, 24, 1 -> SMALL OK | 1/1 | 100.0 | SMALL OK | SMALL OK | 34 -> ICE MISS | 0/1 | BEV OK | BEV OK |
| AUDI_A3_MK3 | Audi A3 | 3.97 | SMALL | unlabelled: MIXED | seen | lab | none 100 | 6 | 127 | 237 | 34.8 | 73, 24, 3 -> SMALL OK | 5/6 | 95.3 | SMALL OK | SMALL OK | 14 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| CHEVROLET_BOLT_EUV | Chevrolet Bolt EUV | 3.99 | SMALL | BEV | seen | lab | none 100 | 160 | 162 | 380 | 43.3 | 67, 23, 10 -> SMALL OK | 142/160 | 88.9 | MID MISS | SMALL OK | 73 -> BEV OK | 118/160 | BEV OK | BEV OK |
| VOLKSWAGEN_GOLF_MK7 | Volkswagen Golf | 4.02 | SMALL | unlabelled: MIXED | seen | lab | none 100 | 40 | 178 | 391 | 46.6 | 73, 24, 3 -> SMALL OK | 38/40 | 92.7 | SMALL OK | SMALL OK | 20 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| ACURA_ILX | Acura ILX | 4.06 | SMALL | ICE | seen | lab | radar 100 | 1 | 3 | 4 | 29.8 | 40, 29, 30 -> SMALL OK | 1/1 | 66.7 | MID MISS | SMALL OK | 4 -> ICE OK | 1/1 | ICE OK | ICE OK |
| TOYOTA_CHR | Toyota C-HR | 4.08 | SMALL | unlabelled: MIXED | seen | lab | none 100 | 7 | 97 | 156 | 55.7 | 74, 23, 3 -> SMALL OK | 7/7 | 88.7 | MID MISS | SMALL OK | 24 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_CHR_TSS2 | Toyota C-HR | 4.08 | SMALL | unlabelled: MIXED | seen | lab | none 100 | 2 | 11 | 17 | 36.4 | 77, 22, 1 -> SMALL OK | 2/2 | 90.9 | MID MISS | SMALL OK | 21 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| SUBARU_IMPREZA | Subaru Impreza | 4.13 | SMALL | ICE | seen | lab | none 100 | 25 | 154 | 358 | 42.2 | 77, 20, 4 -> SMALL OK | 25/25 | 96.8 | SMALL OK | SMALL OK | 15 -> ICE OK | 25/25 | ICE OK | ICE OK |
| SUBARU_IMPREZA_2020 | Subaru Impreza | 4.13 | SMALL | ICE | seen | lab | none 100 | 24 | 168 | 351 | 36.3 | 65, 28, 7 -> SMALL OK | 23/24 | 78.0 | SMALL OK | SMALL OK | 14 -> ICE OK | 23/24 | ICE OK | ICE OK |
| TOYOTA_PRIUS | Toyota Prius | 4.13 | SMALL | unlabelled: HEV | seen | lab | radar 98 / none 2 | 109 | 109 | 368 | 47.6 | 75, 23, 2 -> SMALL OK | 105/109 | 96.3 | SMALL OK | SMALL OK | 41 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_RAV4 | Toyota RAV4 | 4.13 | SMALL | ICE | seen | lab | radar 93 / none 7 | 40 | 192 | 395 | 32.5 | 67, 28, 4 -> SMALL OK | 39/40 | 84.4 | MID MISS | SMALL OK | 20 -> ICE OK | 39/40 | ICE OK | ICE OK |
| TOYOTA_RAV4H | Toyota RAV4 | 4.13 | SMALL | unlabelled: HEV | seen | lab | radar 96 / none 4 | 37 | 166 | 389 | 40.4 | 61, 35, 4 -> SMALL OK | 33/37 | 77.7 | SMALL OK | SMALL OK | 35 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_PRIUS_TSS2 | Toyota Prius | 4.14 | SMALL | unlabelled: HEV | seen | lab | radar 95 / none 5 | 99 | 162 | 399 | 50.3 | 69, 29, 3 -> SMALL OK | 93/99 | 92.0 | SMALL OK | SMALL OK | 47 -> ICE (unlabelled) | - | ICE (unlabelled) | BEV (unlabelled) |
| SKODA_KAROQ_MK1 | Skoda Karoq | 4.15 | SMALL | ICE | seen | lab | none 100 | 3 | 8 | 8 | 16.3 | 86, 11, 3 -> SMALL OK | 3/3 | 100.0 | SMALL OK | SMALL OK | 10 -> ICE OK | 3/3 | ICE OK | ICE OK |
| VOLKSWAGEN_TAOS_MK1 | Volkswagen Taos | 4.18 | SMALL | ICE | seen | lab | none 100 | 2 | 24 | 55 | 30.4 | 75, 23, 2 -> SMALL OK | 2/2 | 100.0 | SMALL OK | SMALL OK | 17 -> ICE OK | 2/2 | ICE OK | ICE OK |
| CHEVROLET_VOLT | Chevrolet Volt | 4.18 | SMALL | unlabelled: PHEV | seen | lab | none 100 | 24 | 203 | 386 | 34.4 | 70, 24, 6 -> SMALL OK | 23/24 | 88.7 | SMALL OK | SMALL OK | 42 -> ICE (unlabelled) | - | BEV (unlabelled) | BEV (unlabelled) |
| SUBARU_FORESTER | Subaru Forester | 4.19 | SMALL | ICE | seen | lab | none 100 | 35 | 210 | 388 | 45.1 | 69, 25, 6 -> SMALL OK | 34/35 | 82.9 | SMALL OK | SMALL OK | 16 -> ICE OK | 34/35 | ICE OK | ICE OK |
| SUBARU_FORESTER_2022 | Subaru Forester | 4.19 | SMALL | ICE | seen | lab | none 100 | 1 | 19 | 49 | 17.2 | 66, 29, 5 -> SMALL OK | 1/1 | 100.0 | SMALL OK | SMALL OK | 8 -> ICE OK | 1/1 | ICE OK | ICE OK |
| CHEVROLET_VOLT_2019 | Chevrolet Volt | 4.19 | SMALL | unlabelled: PHEV | seen | lab | none 100 | 1 | 1 | 5 | 99.8 | 95, 4, 1 -> SMALL OK | 1/1 | 100.0 | SMALL OK | SMALL OK | 66 -> BEV (unlabelled) | - | BEV (unlabelled) | BEV (unlabelled) |
| HONDA_INSIGHT | Honda Insight | 4.20 | SMALL | unlabelled: HEV | seen | lab | none 100 | 15 | 181 | 334 | 51.8 | 72, 26, 3 -> SMALL OK | 14/15 | 92.3 | SMALL OK | SMALL OK | 19 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| LEXUS_NX | Lexus NX | 4.20 | SMALL | ICE | seen | public | radar 99 / none 1 | 9 | 229 | 371 | 26.7 | 70, 28, 2 -> SMALL OK | 8/9 | 89.5 | MID MISS | SMALL OK | 24 -> ICE OK | 9/9 | ICE OK | ICE OK |
| LEXUS_NX_TSS2 | Lexus NX | 4.20 | SMALL | ICE | seen | lab | radar 98 / none 2 | 15 | 144 | 268 | 44.9 | 69, 28, 2 -> SMALL OK | 15/15 | 87.5 | MID MISS | SMALL OK | 28 -> ICE OK | 15/15 | ICE OK | ICE OK |
| FORD_FOCUS_MK4 | Ford Focus | 4.22 | SMALL | ICE | seen | lab | radar 100 | 1 | 9 | 24 | 2.9 | 73, 24, 3 -> SMALL OK | 1/1 | 88.9 | SMALL OK | SMALL OK | 31 -> ICE OK | 1/1 | ICE OK | ICE OK |
| SUBARU_OUTBACK | Subaru Outback | 4.23 | SMALL | ICE | seen | lab | none 100 | 41 | 162 | 385 | 45.7 | 66, 28, 7 -> SMALL OK | 36/41 | 77.8 | SMALL OK | SMALL OK | 17 -> ICE OK | 41/41 | ICE OK | ICE OK |
| AUDI_Q3_MK2 | Audi Q3 | 4.23 | SMALL | ICE | seen | lab | none 100 | 11 | 192 | 380 | 17.8 | 72, 25, 3 -> SMALL OK | 11/11 | 89.1 | SMALL OK | SMALL OK | 16 -> ICE OK | 11/11 | ICE OK | ICE OK |
| SUBARU_LEGACY | Subaru Legacy | 4.24 | SMALL | ICE | seen | lab | none 100 | 2 | 23 | 51 | 34.0 | 75, 20, 5 -> SMALL OK | 2/2 | 91.3 | SMALL OK | SMALL OK | 14 -> ICE OK | 2/2 | ICE OK | ICE OK |
| KIA_NIRO_EV_2ND_GEN | Kia Niro | 4.26 | SMALL | BEV | seen | lab | none 100 | 3 | 59 | 127 | 3.5 | 69, 28, 3 -> SMALL OK | 3/3 | 84.7 | SMALL OK | SMALL OK | 95 -> BEV OK | 3/3 | BEV OK | BEV OK |
| SUBARU_OUTBACK_PREGLOBAL_2018 | Subaru Outback | 4.27 | SMALL | ICE | seen | lab | none 100 | 5 | 46 | 106 | 21.6 | 59, 17, 24 -> SMALL OK | 4/5 | 76.1 | SMALL OK | SMALL OK | 14 -> ICE OK | 5/5 | ICE OK | ICE OK |
| KIA_NIRO_EV | Kia Niro | 4.27 | SMALL | BEV | seen | lab | scc 48 / radar 31 / none 21 | 38 | 180 | 385 | 43.6 | 64, 33, 3 -> SMALL OK | 34/38 | 82.2 | SMALL OK | SMALL OK | 71 -> BEV OK | 33/38 | BEV OK | BEV OK |
| HONDA_CRV_HYBRID | Honda CR-V | 4.28 | SMALL | unlabelled: HEV | seen | lab | none 100 | 13 | 185 | 304 | 41.8 | 68, 27, 5 -> SMALL OK | 13/13 | 89.7 | MID MISS | SMALL OK | 26 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| FORD_BRONCO_SPORT_MK1 | Ford Bronco Sport | 4.28 | SMALL | ICE | seen | lab | radar 100 | 8 | 30 | 94 | 60.5 | 42, 50, 8 -> MID MISS | 3/8 | 46.7 | SMALL OK | SMALL OK | 38 -> ICE OK | 7/8 | ICE OK | BEV MISS |
| KIA_NIRO_HEV_2021 | Kia Niro | 4.29 | SMALL | unlabelled: HEV | seen | lab | scc 78 / none 22 | 4 | 98 | 144 | 48.3 | 73, 24, 2 -> SMALL OK | 4/4 | 89.8 | SMALL OK | SMALL OK | 46 -> ICE (unlabelled) | - | ICE (unlabelled) | BEV (unlabelled) |
| KIA_NIRO_HEV_2ND_GEN | Kia Niro | 4.29 | SMALL | unlabelled: HEV | seen | lab | none 100 | 8 | 116 | 187 | 63.7 | 72, 25, 3 -> SMALL OK | 8/8 | 90.5 | SMALL OK | SMALL OK | 42 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| KIA_NIRO_PHEV | Kia Niro | 4.29 | SMALL | unlabelled: PHEV | seen | lab | none 67 / scc 21 / radar 13 | 3 | 22 | 63 | 28.0 | 69, 27, 4 -> SMALL OK | 3/3 | 95.5 | SMALL OK | SMALL OK | 33 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| KIA_NIRO_PHEV_2022 | Kia Niro | 4.29 | SMALL | unlabelled: PHEV | seen | lab | scc 66 / none 34 | 2 | 74 | 95 | 52.3 | 68, 29, 3 -> SMALL OK | 2/2 | 81.1 | SMALL OK | SMALL OK | 48 -> ICE (unlabelled) | - | ICE (unlabelled) | BEV (unlabelled) |
| MAZDA_CX5 | Mazda CX-5 | 4.31 | SMALL | ICE | seen | lab | none 100 | 1 | 3 | 4 | 0.0 | 78, 19, 3 -> SMALL OK | 1/1 | 100.0 | SMALL OK | SMALL OK | 7 -> ICE OK | 1/1 | ICE OK | ICE OK |
| MAZDA_CX5_2022 | Mazda CX-5 | 4.31 | SMALL | ICE | seen | lab | none 100 | 80 | 136 | 380 | 40.9 | 71, 25, 4 -> SMALL OK | 78/80 | 94.1 | SMALL OK | SMALL OK | 15 -> ICE OK | 77/80 | ICE OK | ICE OK |
| HYUNDAI_ELANTRA_2021 | Hyundai Elantra | 4.31 | SMALL | ICE | seen | lab | scc 71 / none 29 | 12 | 150 | 312 | 40.5 | 68, 28, 4 -> SMALL OK | 12/12 | 90.0 | SMALL OK | SMALL OK | 24 -> ICE OK | 11/12 | ICE OK | ICE OK |
| HYUNDAI_ELANTRA_HEV_2021 | Hyundai Elantra | 4.31 | SMALL | unlabelled: HEV | seen | lab | scc 78 / none 22 | 11 | 224 | 334 | 41.8 | 62, 35, 3 -> SMALL OK | 11/11 | 77.2 | SMALL OK | SMALL OK | 54 -> BEV (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| NISSAN_ROGUE | Nissan Rogue | 4.31 | SMALL | ICE | seen | lab | none 100 | 5 | 65 | 100 | 63.5 | 67, 28, 5 -> SMALL OK | 3/5 | 84.6 | SMALL OK | SMALL OK | 27 -> ICE OK | 5/5 | ICE OK | ICE OK |
| ACURA_RDX | Acura RDX | 4.31 | SMALL | ICE | seen | public | radar 96 / none 4 | 3 | 18 | 144 | 35.9 | 56, 39, 5 -> SMALL OK | 3/3 | 72.2 | MID MISS | LARGE MISS | 25 -> ICE OK | 3/3 | ICE OK | ICE OK |
| TOYOTA_RAV4_PRIME | Toyota RAV4 | 4.32 | SMALL | unlabelled: PHEV | seen | lab | radar 95 / none 5 | 40 | 138 | 384 | 43.3 | 65, 30, 5 -> SMALL OK | 35/40 | 81.2 | SMALL OK | SMALL OK | 56 -> BEV (unlabelled) | - | ICE (unlabelled) | BEV (unlabelled) |
| LEXUS_IS | Lexus IS | 4.32 | SMALL | ICE | seen | lab | radar 97 / none 3 | 8 | 203 | 317 | 16.7 | 65, 30, 5 -> SMALL OK | 8/8 | 82.8 | MID MISS | SMALL OK | 22 -> ICE OK | 8/8 | ICE OK | ICE OK |
| LEXUS_RC | Lexus RC | 4.33 | SMALL | ICE | seen | public | radar 98 / none 2 | 3 | 38 | 52 | 24.5 | 66, 30, 3 -> SMALL OK | 3/3 | 92.1 | MID MISS | MID MISS | 20 -> ICE OK | 2/3 | ICE OK | ICE OK |
| VOLKSWAGEN_TIGUAN_MK2 | Volkswagen Tiguan | 4.33 | SMALL | unlabelled: MIXED | seen | lab | none 100 | 31 | 194 | 385 | 38.2 | 71, 24, 4 -> SMALL OK | 30/31 | 91.2 | SMALL OK | SMALL OK | 17 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_RAV4_TSS2 | Toyota RAV4 | 4.34 | SMALL | ICE | seen | lab | radar 96 / none 4 | 113 | 113 | 384 | 49.8 | 63, 34, 3 -> SMALL OK | 88/113 | 77.9 | SMALL OK | SMALL OK | 44 -> ICE OK | 65/113 | ICE OK | ICE OK |
| TOYOTA_RAV4_TSS2_2022 | Toyota RAV4 | 4.34 | SMALL | unlabelled: HEV | seen | lab | none 100 | 67 | 128 | 384 | 44.5 | 68, 29, 3 -> SMALL OK | 61/67 | 91.4 | SMALL OK | SMALL OK | 32 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_RAV4_TSS2_2023 | Toyota RAV4 | 4.34 | SMALL | ICE | seen | lab | none 100 | 155 | 157 | 386 | 47.4 | 67, 29, 4 -> SMALL OK | 126/155 | 81.5 | SMALL OK | SMALL OK | 34 -> ICE OK | 120/155 | ICE OK | ICE OK |
| KIA_FORTE | Kia Forte | 4.37 | SMALL | ICE | seen | lab | none 57 / scc 43 | 3 | 39 | 76 | 7.7 | 72, 25, 4 -> SMALL OK | 3/3 | 76.9 | SMALL OK | SMALL OK | 20 -> ICE OK | 3/3 | ICE OK | ICE OK |
| FORD_ESCAPE_MK4 | Ford Escape | 4.39 (border) | SMALL | unlabelled: MIXED | seen | lab | radar 81 / none 19 | 13 | 170 | 269 | 42.3 | 57, 33, 9 -> SMALL OK | 11/13 | 67.6 | SMALL OK | LARGE MISS | 39 -> ICE (unlabelled) | - | BEV (unlabelled) | BEV (unlabelled) |
| FORD_ESCAPE_MK4_5 | Ford Escape | 4.39 (border) | SMALL | unlabelled: MIXED | seen | lab | none 100 | 1 | 1 | 1 | 0.0 | 58, 40, 2 -> SMALL OK | 1/1 | 100.0 | MID MISS | LARGE MISS | 3 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| SKODA_KODIAQ_MK1 | Skoda Kodiaq | 4.41 (border) | SMALL | ICE | seen | lab | none 100 | 12 | 166 | 296 | 32.0 | 72, 24, 4 -> SMALL OK | 12/12 | 91.0 | SMALL OK | SMALL OK | 19 -> ICE OK | 12/12 | ICE OK | ICE OK |
| KIA_SPORTAGE_5TH_GEN | Kia Sportage | 4.42 (border) | SMALL | unlabelled: MIXED | seen | lab | none 100 | 9 | 145 | 277 | 39.3 | 65, 31, 4 -> SMALL OK | 9/9 | 82.1 | MID MISS | SMALL OK | 28 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| LEXUS_IS_TSS2 | Lexus IS | 4.46 (border) | MID | ICE | seen | lab | radar 100 / none 0 | 8 | 135 | 241 | 35.7 | 61, 34, 5 -> SMALL MISS | 0/8 | 23.0 | MID OK | MID OK | 35 -> ICE OK | 6/8 | ICE OK | ICE OK |
| CADILLAC_XT4 | Cadillac XT4 | 4.46 (border) | MID | ICE | seen | lab | none 100 | 1 | 1 | 1 | 1.2 | 78, 21, 1 -> SMALL MISS | 0/1 | 0.0 | SMALL MISS | LARGE MISS | 3 -> ICE OK | 1/1 | ICE OK | ICE OK |
| TOYOTA_CAMRY | Toyota Camry | 4.47 (border) | MID | unlabelled: HEV | seen | lab | none 100 | 86 | 164 | 390 | 41.4 | 68, 28, 4 -> SMALL MISS | 4/86 | 11.0 | SMALL MISS | SMALL MISS | 21 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_CAMRY_TSS2 | Toyota Camry | 4.47 (border) | MID | unlabelled: HEV | seen | lab | radar 97 / none 3 | 81 | 155 | 391 | 44.7 | 65, 31, 4 -> SMALL MISS | 8/81 | 12.3 | SMALL MISS | SMALL MISS | 39 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| SKODA_SUPERB_MK3 | Skoda Superb | 4.48 | MID | ICE | seen | lab | none 100 | 10 | 195 | 315 | 42.6 | 66, 30, 4 -> SMALL MISS | 0/10 | 12.3 | SMALL MISS | SMALL MISS | 19 -> ICE OK | 8/10 | ICE OK | ICE OK |
| VOLKSWAGEN_ARTEON_MK1 | Volkswagen Arteon | 4.49 | MID | unlabelled: MIXED | seen | lab | none 100 | 4 | 43 | 134 | 52.3 | 76, 22, 3 -> SMALL MISS | 0/4 | 7.0 | SMALL MISS | SMALL MISS | 28 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| ACURA_RDX_3G | Acura RDX | 4.50 | MID | ICE | seen | lab | none 100 | 18 | 190 | 332 | 34.9 | 66, 30, 4 -> SMALL MISS | 0/18 | 13.2 | MID OK | MID OK | 14 -> ICE OK | 18/18 | ICE OK | ICE OK |
| HYUNDAI_TUCSON_4TH_GEN | Hyundai Tucson | 4.53 | MID | unlabelled: MIXED | seen | lab | none 100 | 27 | 146 | 321 | 41.7 | 64, 31, 4 -> SMALL MISS | 3/27 | 17.8 | MID OK | SMALL MISS | 31 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| HONDA_ACCORD | Honda Accord | 4.54 | MID | unlabelled: HEV | seen | lab | none 100 | 120 | 149 | 390 | 49.0 | 49, 45, 7 -> SMALL MISS | 55/120 | 45.0 | MID OK | SMALL MISS | 19 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| GENESIS_G70_2020 | Genesis G70 | 4.54 | MID | ICE | seen | lab | scc 56 / none 44 | 4 | 41 | 128 | 15.0 | 66, 31, 3 -> SMALL MISS | 0/4 | 43.9 | SMALL MISS | SMALL MISS | 18 -> ICE OK | 4/4 | ICE OK | ICE OK |
| HYUNDAI_SONATA_LF | Hyundai Sonata | 4.56 | MID | ICE | seen | lab | none 69 / scc 31 | 6 | 110 | 220 | 37.9 | 62, 31, 7 -> SMALL MISS | 1/6 | 14.5 | SMALL MISS | SMALL MISS | 19 -> ICE OK | 6/6 | ICE OK | ICE OK |
| TESLA_MODEL_3 | Tesla Model 3 | 4.57 | MID | BEV | seen | lab | none 100 | 28 | 128 | 352 | 52.9 | 42, 53, 5 -> MID OK | 17/28 | 60.9 | MID OK | MID OK | 80 -> BEV OK | 27/28 | BEV OK | BEV OK |
| TOYOTA_HIGHLANDER | Toyota Highlander | 4.60 | MID | ICE | seen | lab | radar 94 / none 6 | 61 | 124 | 387 | 38.8 | 67, 29, 5 -> SMALL MISS | 5/61 | 12.9 | SMALL MISS | SMALL MISS | 32 -> ICE OK | 49/61 | ICE OK | ICE OK |
| LEXUS_ES | Lexus ES | 4.61 | MID | ICE | seen | lab | radar 99 / none 1 | 6 | 119 | 158 | 27.2 | 61, 34, 5 -> SMALL MISS | 0/6 | 26.1 | SMALL MISS | MID OK | 27 -> ICE OK | 6/6 | ICE OK | ICE OK |
| LEXUS_ES_TSS2 | Lexus ES | 4.61 | MID | unlabelled: HEV | seen | lab | radar 97 / none 3 | 62 | 145 | 375 | 42.1 | 51, 43, 6 -> SMALL MISS | 21/62 | 36.6 | MID OK | SMALL MISS | 40 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| HYUNDAI_SONATA | Hyundai Sonata | 4.62 | MID | ICE | seen | lab | scc 52 / radar 31 / none 18 | 37 | 150 | 383 | 54.7 | 63, 31, 6 -> SMALL MISS | 2/37 | 18.7 | SMALL MISS | SMALL MISS | 29 -> ICE OK | 34/37 | ICE OK | ICE OK |
| HYUNDAI_SONATA_HYBRID | Hyundai Sonata | 4.62 | MID | unlabelled: HEV | seen | lab | scc 44 / radar 43 / none 13 | 13 | 174 | 328 | 43.9 | 58, 37, 4 -> SMALL MISS | 1/13 | 30.5 | SMALL MISS | SMALL MISS | 62 -> BEV (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_HIGHLANDER_TSS2 | Toyota Highlander | 4.68 | MID | ICE | seen | lab | radar 96 / none 4 | 164 | 166 | 378 | 37.9 | 61, 34, 5 -> SMALL MISS | 37/164 | 22.3 | MID OK | SMALL MISS | 40 -> ICE OK | 99/164 | ICE OK | ICE OK |
| SUBARU_ASCENT | Subaru Ascent | 4.72 | MID | ICE | seen | lab | none 100 | 34 | 174 | 389 | 40.6 | 73, 21, 6 -> SMALL MISS | 0/34 | 5.7 | SMALL MISS | SMALL MISS | 20 -> ICE OK | 32/34 | ICE OK | ICE OK |
| LEXUS_LC_TSS2 | Lexus LC | 4.74 | MID | ICE | seen | lab | radar 100 | 2 | 53 | 67 | 14.8 | 68, 29, 4 -> SMALL MISS | 0/2 | 9.4 | SMALL MISS | SMALL MISS | 12 -> ICE OK | 2/2 | ICE OK | ICE OK |
| KIA_EV6 | Kia EV6 | 4.74 | MID | BEV | seen | lab | none 100 | 87 | 137 | 375 | 46.0 | 30, 63, 7 -> MID OK | 70/87 | 74.5 | MID OK | SMALL MISS | 78 -> BEV OK | 79/87 | BEV OK | BEV OK |
| TOYOTA_ALPHARD_TSS2 | Toyota Alphard | 4.79 | MID | unlabelled: MIXED | seen | lab | radar 100 | 6 | 82 | 165 | 34.3 | 62, 33, 6 -> SMALL MISS | 0/6 | 17.1 | MID OK | SMALL MISS | 25 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| HYUNDAI_IONIQ_6 | Hyundai Ioniq 6 | 4.80 | MID | BEV | seen | lab | none 100 | 24 | 117 | 280 | 51.9 | 53, 43, 4 -> SMALL MISS | 7/24 | 37.6 | MID OK | SMALL MISS | 74 -> BEV OK | 23/24 | BEV OK | BEV OK |
| HYUNDAI_IONIQ_5 | Hyundai Ioniq 5 | 4.83 | MID | BEV | seen | lab | none 100 | 101 | 122 | 393 | 46.0 | 47, 49, 4 -> MID OK | 46/101 | 46.7 | MID OK | SMALL MISS | 77 -> BEV OK | 89/101 | BEV OK | BEV OK |
| FORD_MUSTANG_MACH_E_MK1 | Ford Mustang Mach-E | 4.84 (border) | MID | BEV | seen | lab | none 100 | 12 | 142 | 305 | 34.2 | 40, 46, 13 -> MID OK | 7/12 | 59.2 | SMALL MISS | SMALL MISS | 80 -> BEV OK | 11/12 | BEV OK | BEV OK |
| HYUNDAI_CUSTIN_1ST_GEN | Hyundai Custin | 4.86 (border) | MID | ICE | seen | lab | scc 82 / none 18 | 4 | 38 | 56 | 40.4 | 29, 24, 47 -> LARGE MISS | 0/4 | 21.1 | LARGE MISS | SMALL MISS | 16 -> ICE OK | 4/4 | ICE OK | ICE OK |
| HYUNDAI_GENESIS | Hyundai Genesis | 4.90 (border) | LARGE | ICE | seen | lab | none 77 / scc 23 | 13 | 165 | 279 | 27.3 | 35, 41, 24 -> MID MISS | 2/13 | 20.6 | MID MISS | SMALL MISS | 23 -> ICE OK | 13/13 | ICE OK | ICE OK |
| VOLKSWAGEN_TRANSPORTER_T61 | Volkswagen Transporter | 4.91 (border) | LARGE | ICE | seen | lab | none 100 | 2 | 50 | 74 | 30.4 | 57, 36, 7 -> SMALL MISS | 0/2 | 2.0 | SMALL MISS | MID MISS | 28 -> ICE OK | 2/2 | ICE OK | ICE OK |
| HONDA_PILOT | Honda Pilot | 4.91 (border) | LARGE | ICE | seen | lab | radar 98 / none 2 | 77 | 165 | 396 | 33.9 | 50, 41, 9 -> SMALL MISS | 2/77 | 4.2 | MID MISS | MID MISS | 24 -> ICE OK | 71/77 | ICE OK | ICE OK |
| FORD_MAVERICK_MK1 | Ford Maverick | 4.93 (border) | LARGE | unlabelled: MIXED | seen | lab | radar 73 / none 27 | 32 | 158 | 382 | 52.3 | 47, 32, 21 -> SMALL MISS | 2/32 | 20.3 | SMALL MISS | SMALL MISS | 32 -> ICE (unlabelled) | - | ICE (unlabelled) | BEV (unlabelled) |
| HYUNDAI_PALISADE | Hyundai Palisade | 4.96 | LARGE | ICE | seen | lab | scc 50 / radar 26 / none 25 | 63 | 173 | 379 | 41.4 | 35, 34, 30 -> SMALL MISS | 24/63 | 34.1 | SMALL MISS | SMALL MISS | 25 -> ICE OK | 56/63 | ICE OK | ICE OK |
| FORD_EXPLORER_MK6 | Ford Explorer | 5.14 | LARGE | unlabelled: MIXED | seen | lab | radar 81 / none 19 | 44 | 173 | 374 | 30.6 | 49, 30, 21 -> SMALL MISS | 5/44 | 22.0 | SMALL MISS | SMALL MISS | 28 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| HONDA_ODYSSEY | Honda Odyssey | 5.14 | LARGE | ICE | seen | lab | radar 99 / none 1 | 96 | 176 | 381 | 28.1 | 56, 36, 8 -> SMALL MISS | 2/96 | 2.8 | SMALL MISS | SMALL MISS | 19 -> ICE OK | 89/96 | ICE OK | ICE OK |
| MAZDA_CX9_2021 | Mazda CX-9 | 5.15 | LARGE | ICE | seen | lab | none 100 | 13 | 155 | 362 | 41.8 | 48, 16, 36 -> SMALL MISS | 4/13 | 43.9 | SMALL MISS | SMALL MISS | 17 -> ICE OK | 13/13 | ICE OK | ICE OK |
| CHEVROLET_TRAVERSE | Chevrolet Traverse | 5.24 | LARGE | ICE | seen | lab | none 100 | 1 | 17 | 29 | 47.5 | 42, 12, 46 -> LARGE OK | 1/1 | 70.6 | LARGE OK | LARGE OK | 16 -> ICE OK | 1/1 | ICE OK | ICE OK |
| HONDA_RIDGELINE | Honda Ridgeline | 5.33 | LARGE | ICE | seen | lab | radar 96 / none 4 | 56 | 154 | 392 | 28.4 | 47, 36, 17 -> SMALL MISS | 2/56 | 11.7 | SMALL MISS | MID MISS | 19 -> ICE OK | 53/56 | ICE OK | ICE OK |
| CHRYSLER_PACIFICA_2018 | Chrysler Pacifica | 5.36 | LARGE | ICE | seen | lab | none 100 | 18 | 174 | 338 | 33.9 | 49, 29, 22 -> SMALL MISS | 2/18 | 22.4 | MID MISS | LARGE OK | 17 -> ICE OK | 17/18 | ICE OK | ICE OK |
| CHRYSLER_PACIFICA_2018_HYBRID | Chrysler Pacifica | 5.36 | LARGE | unlabelled: PHEV | seen | lab | none 100 | 29 | 181 | 378 | 31.4 | 52, 30, 17 -> SMALL MISS | 0/29 | 16.6 | MID MISS | LARGE OK | 36 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| CHRYSLER_PACIFICA_2019_HYBRID | Chrysler Pacifica | 5.36 | LARGE | unlabelled: PHEV | seen | lab | none 100 | 96 | 159 | 388 | 34.7 | 52, 29, 20 -> SMALL MISS | 18/96 | 18.2 | MID MISS | LARGE OK | 40 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| CHRYSLER_PACIFICA_2020 | Chrysler Pacifica | 5.36 | LARGE | ICE | seen | lab | none 100 | 41 | 172 | 371 | 23.4 | 46, 30, 24 -> SMALL MISS | 7/41 | 26.2 | MID MISS | LARGE OK | 17 -> ICE OK | 39/41 | ICE OK | ICE OK |
| RIVIAN_R1 | Rivian R1T/R1S | 5.58 | LARGE | BEV | seen | lab | none 84 / radar 16 | 96 | 96 | 386 | 44.9 | 37, 30, 33 -> SMALL MISS | 39/96 | 40.6 | MID MISS | LARGE OK | 88 -> BEV OK | 92/96 | BEV OK | BEV OK |
| CADILLAC_ESCALADE_ESV_2019 | Cadillac Escalade ESV | 5.76 | LARGE | ICE | seen | public | none 100 | 1 | 2 | 8 | 55.5 | 69, 23, 8 -> SMALL MISS | 0/1 | 0.0 | MID MISS | LARGE OK | 33 -> ICE OK | 1/1 | ICE OK | ICE OK |
| VOLKSWAGEN_CRAFTER_MK2 | Volkswagen Crafter | 6.24 | LARGE | unlabelled: MIXED | seen | lab | none 100 | 4 | 21 | 115 | 42.4 | 49, 23, 28 -> SMALL MISS | 0/4 | 42.9 | SMALL MISS | LARGE OK | 25 -> ICE (unlabelled) | - | BEV (unlabelled) | ICE (unlabelled) |
| FORD_F_150_MK14 | Ford F-150 | 6.44 | LARGE | unlabelled: MIXED | seen | lab | none 100 | 42 | 116 | 387 | 45.8 | 31, 25, 44 -> LARGE OK | 25/42 | 60.3 | SMALL MISS | LARGE OK | 21 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| RAM_HD_5TH_GEN | RAM 2500/3500 HD | 6.57 | LARGE | ICE | seen | lab | none 100 | 5 | 19 | 74 | 19.8 | 40, 11, 49 -> LARGE OK | 4/5 | 63.2 | LARGE OK | LARGE OK | 18 -> ICE OK | 5/5 | ICE OK | ICE OK |
| CHEVROLET_SILVERADO | Chevrolet Silverado 1500 | 6.65 | LARGE | ICE | seen | lab | none 100 | 52 | 137 | 395 | 48.3 | 36, 23, 42 -> LARGE OK | 31/52 | 53.3 | LARGE OK | LARGE OK | 21 -> ICE OK | 50/52 | ICE OK | ICE OK |
| RAM_1500_5TH_GEN | RAM 1500 | 6.73 | LARGE | ICE | seen | lab | none 100 | 99 | 131 | 384 | 36.6 | 40, 14, 47 -> LARGE OK | 57/99 | 58.8 | LARGE OK | LARGE OK | 17 -> ICE OK | 92/99 | ICE OK | ICE OK |
| HYUNDAI_KONA_2022 | Hyundai Kona | 4.02 | SMALL | ICE | held-out | lab | none 100 | 2 | 2 | 2 | 23.2 | 80, 16, 4 -> SMALL OK | 2/2 | 100.0 | SMALL OK | SMALL OK | 2 -> ICE OK | 2/2 | ICE OK | ICE OK |
| HYUNDAI_KONA_EV | Hyundai Kona | 4.02 | SMALL | BEV | held-out | lab | scc 73 / none 27 | 14 | 164 | 387 | 40.4 | 57, 40, 3 -> SMALL OK | 12/14 | 69.5 | SMALL OK | SMALL OK | 77 -> BEV OK | 12/14 | BEV OK | BEV OK |
| HYUNDAI_KONA_EV_2022 | Hyundai Kona | 4.02 | SMALL | BEV | held-out | lab | none 86 / scc 14 | 4 | 90 | 149 | 63.5 | 76, 22, 2 -> SMALL OK | 4/4 | 91.1 | SMALL OK | SMALL OK | 70 -> BEV OK | 4/4 | BEV OK | BEV OK |
| TOYOTA_COROLLA_TSS2 | Toyota Corolla | 4.09 | SMALL | unlabelled: HEV | held-out | lab | radar 97 / none 3 | 233 | 377 | 971 | 50.1 | 68, 28, 3 -> SMALL OK | 223/233 | 89.7 | SMALL OK | SMALL OK | 34 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_COROLLA | Toyota Corolla | 4.14 | SMALL | ICE | held-out | lab | radar 96 / none 4 | 39 | 498 | 981 | 36.8 | 62, 28, 9 -> SMALL OK | 36/39 | 78.7 | SMALL OK | SMALL OK | 19 -> ICE OK | 39/39 | ICE OK | ICE OK |
| NISSAN_LEAF | Nissan Leaf | 4.16 | SMALL | BEV | held-out | lab | none 100 | 23 | 482 | 801 | 32.5 | 53, 44, 3 -> SMALL OK | 13/23 | 58.3 | SMALL OK | SMALL OK | 63 -> BEV OK | 18/23 | ICE MISS | ICE MISS |
| NISSAN_LEAF_IC | Nissan Leaf | 4.16 | SMALL | BEV | held-out | lab | none 100 | 5 | 88 | 115 | 46.2 | 57, 40, 2 -> SMALL OK | 4/5 | 63.6 | SMALL OK | SMALL OK | 63 -> BEV OK | 4/5 | BEV OK | ICE MISS |
| VOLKSWAGEN_JETTA_MK7 | Volkswagen Jetta | 4.17 | SMALL | ICE | held-out | lab | none 100 | 12 | 183 | 342 | 27.0 | 72, 24, 4 -> SMALL OK | 12/12 | 89.6 | SMALL OK | SMALL OK | 12 -> ICE OK | 12/12 | ICE OK | ICE OK |
| HYUNDAI_IONIQ_EV_2020 | Hyundai Ioniq | 4.21 | SMALL | BEV | held-out | lab | scc 66 / none 34 | 5 | 120 | 235 | 61.9 | 70, 27, 3 -> SMALL OK | 5/5 | 85.0 | SMALL OK | SMALL OK | 83 -> BEV OK | 5/5 | BEV OK | BEV OK |
| HYUNDAI_IONIQ_PHEV | Hyundai Ioniq | 4.22 | SMALL | unlabelled: PHEV | held-out | lab | scc 72 / none 28 | 9 | 162 | 292 | 43.2 | 73, 24, 2 -> SMALL OK | 9/9 | 88.9 | SMALL OK | SMALL OK | 50 -> BEV (unlabelled) | - | ICE (unlabelled) | BEV (unlabelled) |
| HYUNDAI_IONIQ | Hyundai Ioniq | 4.24 | SMALL | unlabelled: HEV | held-out | lab | none 75 / scc 25 | 4 | 46 | 104 | 36.9 | 70, 27, 4 -> SMALL OK | 4/4 | 84.8 | SMALL OK | SMALL OK | 30 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| HYUNDAI_IONIQ_HEV_2022 | Hyundai Ioniq | 4.24 | SMALL | unlabelled: HEV | held-out | lab | scc 69 / none 31 | 8 | 159 | 304 | 49.0 | 67, 29, 3 -> SMALL OK | 8/8 | 87.4 | SMALL OK | SMALL OK | 50 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| CHEVROLET_EQUINOX | Chevrolet Equinox | 4.30 | SMALL | ICE | held-out | lab | none 100 | 5 | 165 | 243 | 45.7 | 64, 31, 5 -> SMALL OK | 5/5 | 77.6 | MID MISS | SMALL OK | 20 -> ICE OK | 5/5 | ICE OK | ICE OK |
| VOLKSWAGEN_PASSAT_MK8 | Volkswagen Passat | 4.40 (border) | SMALL | unlabelled: MIXED | held-out | lab | none 100 | 8 | 125 | 256 | 46.2 | 68, 28, 4 -> SMALL OK | 8/8 | 84.0 | SMALL OK | SMALL OK | 21 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| JEEP_GRAND_CHEROKEE | Jeep Grand Cherokee | 4.41 (border) | SMALL | ICE | held-out | lab | none 100 | 21 | 435 | 803 | 31.1 | 50, 40, 10 -> SMALL OK | 16/21 | 57.2 | MID MISS | LARGE MISS | 14 -> ICE OK | 21/21 | ICE OK | ICE OK |
| JEEP_GRAND_CHEROKEE_2019 | Jeep Grand Cherokee | 4.41 (border) | SMALL | ICE | held-out | lab | none 100 | 15 | 371 | 635 | 31.9 | 47, 38, 15 -> SMALL OK | 10/15 | 52.8 | MID MISS | LARGE MISS | 17 -> ICE OK | 15/15 | ICE OK | ICE OK |
| TOYOTA_AVALON | Toyota Avalon | 4.47 (border) | MID | ICE | held-out | lab | radar 99 / none 1 | 6 | 128 | 239 | 15.1 | 73, 24, 4 -> SMALL MISS | 0/6 | 7.0 | SMALL MISS | MID OK | 18 -> ICE OK | 6/6 | ICE OK | ICE OK |
| KIA_STINGER | Kia Stinger | 4.51 | MID | ICE | held-out | lab | scc 61 / none 39 | 8 | 160 | 381 | 49.5 | 37, 54, 9 -> MID OK | 8/8 | 58.1 | MID OK | SMALL MISS | 31 -> ICE OK | 8/8 | ICE OK | ICE OK |
| KIA_STINGER_2022 | Kia Stinger | 4.51 | MID | ICE | held-out | public | scc 77 / none 23 | 2 | 27 | 98 | 27.1 | 30, 62, 9 -> MID OK | 2/2 | 77.8 | MID OK | SMALL MISS | 26 -> ICE OK | 2/2 | ICE OK | ICE OK |
| TOYOTA_AVALON_2019 | Toyota Avalon | 4.53 | MID | unlabelled: HEV | held-out | lab | radar 97 / none 3 | 25 | 462 | 774 | 36.0 | 65, 31, 3 -> SMALL MISS | 1/25 | 16.9 | MID OK | SMALL MISS | 38 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_AVALON_TSS2 | Toyota Avalon | 4.53 | MID | unlabelled: unknown | held-out | public | radar 97 / none 3 | 6 | 162 | 237 | 40.8 | 66, 29, 5 -> SMALL MISS | 0/6 | 14.8 | SMALL MISS | SMALL MISS | 24 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| KIA_OPTIMA_G4_FL | Kia Optima | 4.53 | MID | ICE | held-out | lab | scc 50 / none 50 | 5 | 71 | 163 | 52.7 | 67, 30, 3 -> SMALL MISS | 0/5 | 19.7 | SMALL MISS | SMALL MISS | 30 -> ICE OK | 5/5 | ICE OK | ICE OK |
| HYUNDAI_SANTA_FE | Hyundai Santa Fe | 4.53 | MID | ICE | held-out | lab | scc 46 / radar 36 / none 18 | 24 | 478 | 855 | 43.4 | 44, 47, 9 -> MID OK | 16/24 | 47.3 | MID OK | SMALL MISS | 27 -> ICE OK | 22/24 | ICE OK | ICE OK |
| KIA_SORENTO | Kia Sorento | 4.54 | MID | ICE | held-out | public | none 100 | 1 | 1 | 10 | 0.0 | 52, 45, 3 -> SMALL MISS | 0/1 | 0.0 | SMALL MISS | SMALL MISS | 11 -> ICE OK | 1/1 | ICE OK | ICE OK |
| LEXUS_RX | Lexus RX | 4.55 | MID | unlabelled: HEV | held-out | lab | radar 98 / none 2 | 55 | 447 | 968 | 29.0 | 63, 33, 4 -> SMALL MISS | 4/55 | 21.0 | MID OK | SMALL MISS | 23 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| LEXUS_RX_TSS2 | Lexus RX | 4.55 | MID | ICE | held-out | lab | radar 98 / none 2 | 69 | 436 | 963 | 31.4 | 65, 32, 4 -> SMALL MISS | 3/69 | 19.7 | SMALL MISS | SMALL MISS | 31 -> ICE OK | 60/69 | ICE OK | ICE OK |
| NISSAN_ALTIMA | Nissan Altima | 4.56 | MID | ICE | held-out | lab | none 100 | 4 | 105 | 193 | 35.1 | 58, 32, 10 -> SMALL MISS | 0/4 | 17.1 | SMALL MISS | SMALL MISS | 17 -> ICE OK | 4/4 | ICE OK | ICE OK |
| HYUNDAI_SANTA_FE_2022 | Hyundai Santa Fe | 4.57 | MID | ICE | held-out | lab | scc 69 / none 31 | 22 | 474 | 937 | 35.6 | 43, 45, 12 -> MID OK | 13/22 | 51.9 | MID OK | SMALL MISS | 28 -> ICE OK | 21/22 | ICE OK | ICE OK |
| HYUNDAI_SANTA_FE_HEV_2022 | Hyundai Santa Fe | 4.57 | MID | unlabelled: HEV | held-out | lab | none 57 / scc 43 | 8 | 131 | 262 | 40.0 | 50, 42, 8 -> SMALL MISS | 4/8 | 35.9 | SMALL MISS | SMALL MISS | 47 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| HYUNDAI_SANTA_FE_PHEV_2022 | Hyundai Santa Fe | 4.57 | MID | unlabelled: PHEV | held-out | lab | scc 69 / none 31 | 8 | 139 | 301 | 47.0 | 44, 50, 6 -> MID OK | 6/8 | 51.1 | MID OK | SMALL MISS | 51 -> BEV (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| KIA_K5_2021 | Kia K5 | 4.60 | MID | ICE | held-out | lab | scc 74 / none 26 | 3 | 67 | 99 | 37.8 | 71, 26, 3 -> SMALL MISS | 0/3 | 7.5 | SMALL MISS | SMALL MISS | 23 -> ICE OK | 3/3 | ICE OK | ICE OK |
| HYUNDAI_AZERA_HEV_6TH_GEN | Hyundai Azera | 4.63 | MID | unlabelled: HEV | held-out | lab | none 60 / scc 40 | 1 | 5 | 5 | 79.9 | 63, 32, 6 -> SMALL MISS | 0/1 | 40.0 | MID OK | SMALL MISS | 66 -> BEV (unlabelled) | - | ICE (unlabelled) | BEV (unlabelled) |
| KIA_SORENTO_4TH_GEN | Kia Sorento | 4.64 | MID | ICE | held-out | lab | none 100 | 8 | 161 | 248 | 38.3 | 71, 25, 4 -> SMALL MISS | 0/8 | 10.6 | SMALL MISS | SMALL MISS | 18 -> ICE OK | 8/8 | ICE OK | ICE OK |
| KIA_SORENTO_HEV_4TH_GEN | Kia Sorento | 4.64 | MID | unlabelled: HEV | held-out | lab | none 100 | 8 | 175 | 287 | 21.5 | 68, 27, 5 -> SMALL MISS | 0/8 | 8.0 | SMALL MISS | SMALL MISS | 27 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TESLA_MODEL_Y | Tesla Model Y | 4.73 | MID | BEV | held-out | lab | none 100 | 21 | 271 | 538 | 42.5 | 34, 59, 6 -> MID OK | 17/21 | 66.8 | MID OK | MID OK | 84 -> BEV OK | 20/21 | BEV OK | BEV OK |
| GENESIS_GV70_1ST_GEN | Genesis GV70 | 4.73 | MID | ICE | held-out | lab | none 100 | 8 | 102 | 200 | 44.1 | 57, 36, 7 -> SMALL MISS | 1/8 | 21.6 | MID OK | MID OK | 21 -> ICE OK | 8/8 | ICE OK | ICE OK |
| GENESIS_GV60_EV_1ST_GEN | Genesis GV60 | 4.73 | MID | BEV | held-out | lab | none 100 | 2 | 43 | 97 | 44.7 | 35, 52, 13 -> MID OK | 1/2 | 48.8 | MID OK | SMALL MISS | 78 -> BEV OK | 2/2 | BEV OK | BEV OK |
| HYUNDAI_SANTA_CRUZ_1ST_GEN | Hyundai Santa Cruz | 4.94 | LARGE | ICE | held-out | lab | none 100 | 1 | 45 | 49 | 45.0 | 58, 32, 9 -> SMALL MISS | 0/1 | 6.7 | MID MISS | SMALL MISS | 16 -> ICE OK | 1/1 | ICE OK | ICE OK |
| GENESIS_G80 | Genesis G80 | 4.95 | LARGE | ICE | held-out | public | none 59 / scc 41 | 7 | 148 | 304 | 45.1 | 38, 36, 26 -> SMALL MISS | 1/7 | 27.0 | SMALL MISS | SMALL MISS | 27 -> ICE OK | 7/7 | ICE OK | ICE OK |
| GENESIS_GV80 | Genesis GV80 | 4.96 | LARGE | ICE | held-out | lab | none 100 | 3 | 93 | 146 | 10.3 | 72, 22, 6 -> SMALL MISS | 0/3 | 2.2 | SMALL MISS | MID MISS | 11 -> ICE OK | 3/3 | ICE OK | ICE OK |
| VOLKSWAGEN_ATLAS_MK1 | Volkswagen Atlas | 5.11 | LARGE | ICE | held-out | lab | none 100 | 32 | 503 | 961 | 38.0 | 54, 37, 9 -> SMALL MISS | 0/32 | 5.6 | SMALL MISS | SMALL MISS | 14 -> ICE OK | 32/32 | ICE OK | ICE OK |
| GENESIS_G90 | Genesis G90 | 5.18 | LARGE | ICE | held-out | lab | none 59 / scc 41 | 9 | 125 | 249 | 48.7 | 53, 41, 6 -> SMALL MISS | 0/9 | 2.4 | MID MISS | MID MISS | 27 -> ICE OK | 9/9 | ICE OK | ICE OK |
| TOYOTA_SIENNA | Toyota Sienna | 5.21 | LARGE | ICE | held-out | lab | radar 96 / none 4 | 62 | 499 | 975 | 30.5 | 61, 34, 5 -> SMALL MISS | 0/62 | 1.0 | SMALL MISS | SMALL MISS | 25 -> ICE OK | 58/62 | ICE OK | ICE OK |
| TOYOTA_SIENNA_4TH_GEN | Toyota Sienna | 5.30 | LARGE | unlabelled: HEV | held-out | lab | radar 96 / none 4 | 32 | 412 | 879 | 40.6 | 39, 28, 34 -> SMALL MISS | 15/32 | 45.1 | SMALL MISS | SMALL MISS | 36 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| FORD_RANGER_MK2 | Ford Ranger | 5.30 | LARGE | ICE | held-out | lab | none 100 | 3 | 36 | 82 | 51.0 | 43, 19, 38 -> SMALL MISS | 2/3 | 63.9 | SMALL MISS | LARGE OK | 33 -> ICE OK | 3/3 | ICE OK | ICE OK |
| KIA_CARNIVAL_4TH_GEN | Kia Carnival | 5.30 | LARGE | ICE | held-out | lab | none 100 | 15 | 291 | 477 | 36.6 | 55, 38, 7 -> SMALL MISS | 0/15 | 2.7 | MID MISS | SMALL MISS | 20 -> ICE OK | 15/15 | ICE OK | ICE OK |
| FORD_F_150_LIGHTNING_MK1 | Ford F-150 Lightning | 6.41 | LARGE | BEV | held-out | lab | none 100 | 6 | 33 | 113 | 28.2 | 26, 20, 55 -> LARGE OK | 5/6 | 84.8 | SMALL MISS | LARGE OK | 52 -> BEV OK | 4/6 | BEV OK | BEV OK |
| LEXUS_CTH | Lexus CT 200h | 3.98 | SMALL | unlabelled: HEV | absent | public | radar 98 / none 2 | 3 | 42 | 65 | 23.8 | 55, 34, 11 -> SMALL OK | 3/3 | 64.3 | SMALL OK | SMALL OK | 35 -> ICE (unlabelled) | - | ICE (unlabelled) | BEV (unlabelled) |
| VOLKSWAGEN_TROC_MK1 | Volkswagen T-Roc | 3.99 | SMALL | ICE | absent | public | none 100 | 3 | 82 | 147 | 24.7 | 78, 16, 5 -> SMALL OK | 3/3 | 92.7 | SMALL OK | SMALL OK | 16 -> ICE OK | 3/3 | ICE OK | ICE OK |
| SKODA_KAMIQ_MK1 | Skoda Kamiq | 4.04 | SMALL | ICE | absent | public | none 100 | 1 | 9 | 35 | 88.1 | 86, 11, 2 -> SMALL OK | 1/1 | 100.0 | SMALL OK | SMALL OK | 23 -> ICE OK | 1/1 | ICE OK | ICE OK |
| SEAT_ATECA_MK1 | SEAT Ateca | 4.11 | SMALL | ICE | absent | public | none 100 | 4 | 79 | 152 | 50.8 | 75, 23, 2 -> SMALL OK | 4/4 | 96.2 | SMALL OK | SMALL OK | 18 -> ICE OK | 4/4 | ICE OK | ICE OK |
| HYUNDAI_ELANTRA_GT_I30 | Hyundai Elantra GT / i30 | 4.13 | SMALL | ICE | absent | public | scc 83 / none 17 | 2 | 51 | 100 | 67.7 | 69, 28, 3 -> SMALL OK | 2/2 | 88.2 | SMALL OK | SMALL OK | 42 -> ICE OK | 2/2 | ICE OK | ICE OK |
| SKODA_OCTAVIA_MK3 | Skoda Octavia | 4.15 | SMALL | ICE | absent | public | none 100 | 7 | 130 | 286 | 39.2 | 77, 21, 2 -> SMALL OK | 7/7 | 91.5 | SMALL OK | SMALL OK | 19 -> ICE OK | 7/7 | ICE OK | ICE OK |
| HYUNDAI_IONIQ_PHEV_2019 | Hyundai Ioniq Plug-in | 4.22 | SMALL | unlabelled: PHEV | absent | public | none 72 / scc 28 | 2 | 35 | 100 | 43.2 | 70, 23, 7 -> SMALL OK | 2/2 | 88.6 | SMALL OK | SMALL OK | 42 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_MIRAI | Toyota Mirai | 4.28 | SMALL | unlabelled: FCEV | absent | public | radar 100 | 6 | 140 | 296 | 27.6 | 66, 30, 4 -> SMALL OK | 6/6 | 84.3 | MID MISS | SMALL OK | 41 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| TOYOTA_PRIUS_V | Toyota Prius v | 4.29 | SMALL | unlabelled: HEV | absent | public | radar 100 | 3 | 17 | 50 | 56.1 | 63, 32, 5 -> SMALL OK | 3/3 | 76.5 | SMALL OK | SMALL OK | 57 -> BEV (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |
| VOLKSWAGEN_TOURAN_MK2 | Volkswagen Touran | 4.39 (border) | SMALL | ICE | absent | lab | none 100 | 1 | 28 | 44 | 9.2 | 70, 25, 5 -> SMALL OK | 1/1 | 96.4 | SMALL OK | SMALL OK | 23 -> ICE OK | 1/1 | ICE OK | ICE OK |
| BUICK_REGAL | Buick Regal Sportback | 4.52 | MID | ICE | absent | public | none 100 | 1 | 33 | 50 | 48.9 | 61, 32, 7 -> SMALL MISS | 0/1 | 18.2 | SMALL MISS | MID OK | 22 -> ICE OK | 1/1 | ICE OK | ICE OK |
| DODGE_DURANGO | Dodge Durango | 4.95 | LARGE | ICE | absent | public | none 100 | 1 | 1 | 11 | 0.0 | 59, 20, 22 -> SMALL MISS | 0/1 | 0.0 | SMALL MISS | LARGE OK | 16 -> ICE OK | 1/1 | ICE OK | ICE OK |
| CADILLAC_ESCALADE | Cadillac Escalade | 5.14 | LARGE | ICE | absent | public | none 100 | 1 | 1 | 1 | 99.8 | 65, 16, 19 -> SMALL MISS | 0/1 | 0.0 | SMALL MISS | LARGE OK | 93 -> BEV MISS | 0/1 | ICE OK | ICE OK |
| HYUNDAI_STARIA_4TH_GEN | Hyundai Staria | 5.65 | LARGE | ICE | absent | public | none 100 | 1 | 15 | 49 | 31.6 | 64, 31, 4 -> SMALL MISS | 0/1 | 0.0 | MID MISS | LARGE OK | 32 -> ICE OK | 1/1 | ICE OK | ICE OK |

### commaCarSegments (`ccs`): pooled per stratum, every run (window / drive / vehicle = device / model = car model)

Accuracy and macro-F1 in percent; the vehicle and model levels carry 95 % intervals from a vehicle-cluster bootstrap of 300 resamples over the stratum's devices (score.py `level_eval` on the stratum subset; the pooled score files use 1,000). `units right per class` = diagonal of the confusion matrix / support; chance = 1 / K and the majority share of the level. Runs: full = the 20-channel models; V1 / V2 / V3 = the reduced-input variants; full+mask-Vk = the full models with that preset masked; full+mask-yaw / full+mask-lead = the sensitivity runs (steering-derived yaw group masked; lead group masked).

size (SMALL / MID / LARGE):

| run | stratum | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | units right per class | chance uniform / majority % |
|---|---|---|---|---|---|---|---|---|
| full | all | window | 45763 | 49.6 | 43.2 | SMALL 78.1 / MID 32.6 / LARGE 20.1 | SMALL 15654/20036 / MID 4943/15159 / LARGE 2120/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 51.3 | 44.0 | SMALL 81.6 / MID 29.7 / LARGE 21.3 | SMALL 8091/9913 / MID 2162/7278 / LARGE 1023/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 54.1 [52.7, 55.6] | 46.4 [44.9, 48.2] | SMALL 91.1 / MID 27.6 / LARGE 23.5 | SMALL 1606/1762 / MID 351/1274 / LARGE 241/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 55.8 [53.0, 58.7] | 45.6 [41.2, 49.8] | SMALL 98.0 / MID 24.2 / LARGE 20.0 | SMALL 49/50 / MID 8/33 / LARGE 6/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 53.3 | 46.1 | SMALL 81.3 / MID 31.2 / LARGE 24.8 | SMALL 9868/12141 / MID 2264/7254 / LARGE 1557/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 56.6 | 47.5 | SMALL 86.0 / MID 27.1 / LARGE 26.9 | SMALL 5019/5833 / MID 867/3200 / LARGE 697/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 53.9 [52.1, 55.7] | 47.3 [45.3, 49.2] | SMALL 90.8 / MID 28.3 / LARGE 25.8 | SMALL 1220/1343 / MID 281/992 / LARGE 223/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 57.1 [52.9, 60.7] | 47.1 [40.7, 51.6] | SMALL 96.9 / MID 20.0 / LARGE 27.8 | SMALL 31/32 / MID 4/20 / LARGE 5/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 42.3 | 37.1 | SMALL 71.0 / MID 33.9 / LARGE 13.2 | SMALL 4700/6620 / MID 2663/7855 / LARGE 560/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 42.8 | 37.8 | SMALL 73.0 / MID 31.9 / LARGE 14.9 | SMALL 2531/3467 / MID 1289/4045 / LARGE 326/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 53.8 [50.9, 57.4] | 41.6 [37.9, 45.0] | SMALL 91.1 / MID 25.6 / LARGE 13.2 | SMALL 368/404 / MID 75/293 / LARGE 22/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 44.8 [40.0, 51.9] | 40.0 [33.3, 49.7] | SMALL 100.0 / MID 33.3 / LARGE 11.1 | SMALL 8/8 / MID 4/12 / LARGE 1/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 79.7 | 36.5 | SMALL 85.2 / MID 32.0 / LARGE 4.9 | SMALL 1086/1275 / MID 16/50 / LARGE 3/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 82.5 | 33.8 | SMALL 88.3 / MID 18.2 / LARGE 0.0 | SMALL 541/613 / MID 6/33 / LARGE 0/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 88.9 [77.8, 97.2] | 31.4 [29.2, 49.3] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 32/32 / MID 0/1 / LARGE 0/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 71.4 [63.6, 90.9] | 27.8 [25.9, 47.5] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 10/10 / MID 0/1 / LARGE 0/3 | 33.3 / 71.4 |
| V1 | all | window | 45763 | 47.5 | 42.3 | SMALL 64.2 / MID 47.1 / LARGE 16.4 | SMALL 12872/20036 / MID 7141/15159 / LARGE 1734/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 48.5 | 42.7 | SMALL 65.1 / MID 47.4 / LARGE 16.0 | SMALL 6451/9913 / MID 3451/7278 / LARGE 769/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 48.2 [46.8, 49.6] | 44.2 [42.6, 45.8] | SMALL 60.7 / MID 54.6 / LARGE 18.8 | SMALL 1069/1762 / MID 695/1274 / LARGE 193/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 54.0 [50.5, 57.1] | 46.2 [42.3, 49.3] | SMALL 78.0 / MID 54.5 / LARGE 13.3 | SMALL 39/50 / MID 18/33 / LARGE 4/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 48.7 | 44.4 | SMALL 63.8 / MID 47.1 / LARGE 21.3 | SMALL 7740/12141 / MID 3414/7254 / LARGE 1339/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 49.9 | 44.7 | SMALL 64.9 / MID 45.8 / LARGE 21.3 | SMALL 3786/5833 / MID 1467/3200 / LARGE 553/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 48.1 [46.5, 49.9] | 45.1 [43.4, 47.0] | SMALL 60.8 / MID 54.2 / LARGE 21.4 | SMALL 817/1343 / MID 538/992 / LARGE 185/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 54.3 [50.0, 58.6] | 48.7 [44.3, 52.7] | SMALL 75.0 / MID 50.0 / LARGE 22.2 | SMALL 24/32 / MID 10/20 / LARGE 4/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 44.3 | 37.8 | SMALL 63.4 / MID 47.2 / LARGE 9.3 | SMALL 4196/6620 / MID 3707/7855 / LARGE 393/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 45.3 | 38.8 | SMALL 63.6 / MID 48.8 / LARGE 9.9 | SMALL 2206/3467 / MID 1973/4045 / LARGE 216/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 47.8 [44.8, 51.3] | 38.8 [35.7, 42.1] | SMALL 58.4 / MID 56.3 / LARGE 7.2 | SMALL 236/404 / MID 165/293 / LARGE 12/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 48.3 [37.9, 50.9] | 38.0 [30.1, 39.6] | SMALL 75.0 / MID 66.7 / LARGE 0.0 | SMALL 6/8 / MID 8/12 / LARGE 0/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 69.1 | 32.5 | SMALL 73.4 / MID 40.0 / LARGE 3.3 | SMALL 936/1275 / MID 20/50 / LARGE 2/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 70.9 | 31.4 | SMALL 74.9 / MID 33.3 / LARGE 0.0 | SMALL 459/613 / MID 11/33 / LARGE 0/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 75.0 [61.1, 87.6] | 29.0 [25.4, 46.9] | SMALL 84.4 / MID 0.0 / LARGE 0.0 | SMALL 27/32 / MID 0/1 / LARGE 0/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 64.3 [58.3, 89.5] | 27.3 [25.3, 47.4] | SMALL 90.0 / MID 0.0 / LARGE 0.0 | SMALL 9/10 / MID 0/1 / LARGE 0/3 | 33.3 / 71.4 |
| V2 | all | window | 45763 | 40.6 | 37.9 | SMALL 54.1 / MID 29.7 / LARGE 30.5 | SMALL 10845/20036 / MID 4501/15159 / LARGE 3225/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 42.1 | 38.4 | SMALL 58.4 / MID 28.4 / LARGE 29.2 | SMALL 5789/9913 / MID 2068/7278 / LARGE 1402/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 42.0 [40.5, 43.5] | 38.9 [37.4, 40.4] | SMALL 59.4 / MID 27.9 / LARGE 29.6 | SMALL 1046/1762 / MID 356/1274 / LARGE 303/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 50.4 [45.9, 54.3] | 45.4 [39.2, 50.4] | SMALL 78.0 / MID 21.2 / LARGE 36.7 | SMALL 39/50 / MID 7/33 / LARGE 11/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 42.0 | 38.6 | SMALL 55.3 / MID 26.8 / LARGE 33.7 | SMALL 6719/12141 / MID 1942/7254 / LARGE 2114/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 44.5 | 39.5 | SMALL 59.9 / MID 25.7 / LARGE 32.9 | SMALL 3494/5833 / MID 823/3200 / LARGE 853/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 42.8 [41.4, 44.5] | 39.5 [38.0, 41.3] | SMALL 63.2 / MID 24.4 / LARGE 32.1 | SMALL 849/1343 / MID 242/992 / LARGE 277/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 54.3 [47.1, 58.2] | 49.9 [40.4, 54.9] | SMALL 78.1 / MID 25.0 / LARGE 44.4 | SMALL 25/32 / MID 5/20 / LARGE 8/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 37.7 | 35.9 | SMALL 51.5 / MID 32.5 / LARGE 25.6 | SMALL 3409/6620 / MID 2555/7855 / LARGE 1086/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 38.3 | 36.1 | SMALL 55.8 / MID 30.7 / LARGE 24.8 | SMALL 1934/3467 / MID 1242/4045 / LARGE 541/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 38.0 [34.7, 41.3] | 33.8 [30.8, 37.8] | SMALL 45.0 / MID 40.6 / LARGE 16.2 | SMALL 182/404 / MID 119/293 / LARGE 27/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 34.5 [32.1, 45.5] | 29.6 [22.9, 43.0] | SMALL 87.5 / MID 16.7 / LARGE 11.1 | SMALL 7/8 / MID 2/12 / LARGE 1/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 53.8 | 28.6 | SMALL 56.2 / MID 8.0 / LARGE 41.0 | SMALL 717/1275 / MID 4/50 / LARGE 25/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 56.1 | 28.7 | SMALL 58.9 / MID 9.1 / LARGE 47.1 | SMALL 361/613 / MID 3/33 / LARGE 8/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 63.9 [50.0, 77.8] | 44.5 [26.0, 82.7] | SMALL 65.6 / MID 0.0 / LARGE 66.7 | SMALL 21/32 / MID 0/1 / LARGE 2/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 64.3 [50.0, 80.0] | 46.8 [23.5, 91.2] | SMALL 70.0 / MID 0.0 / LARGE 66.7 | SMALL 7/10 / MID 0/1 / LARGE 2/3 | 33.3 / 71.4 |
| V3 | all | window | 45763 | 44.7 | 41.7 | SMALL 61.7 / MID 32.2 / LARGE 30.6 | SMALL 12362/20036 / MID 4878/15159 / LARGE 3233/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 45.7 | 41.4 | SMALL 65.6 / MID 30.1 / LARGE 28.2 | SMALL 6502/9913 / MID 2192/7278 / LARGE 1351/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 53.7 [52.2, 55.3] | 50.0 [48.5, 51.5] | SMALL 80.5 / MID 24.0 / LARGE 44.5 | SMALL 1419/1762 / MID 306/1274 / LARGE 456/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 57.5 [53.3, 59.8] | 50.2 [44.4, 53.6] | SMALL 92.0 / MID 15.2 / LARGE 46.7 | SMALL 46/50 / MID 5/33 / LARGE 14/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 48.9 | 46.4 | SMALL 62.1 / MID 30.9 / LARGE 44.1 | SMALL 7534/12141 / MID 2243/7254 / LARGE 2766/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 51.6 | 47.9 | SMALL 66.9 / MID 29.4 / LARGE 44.4 | SMALL 3903/5833 / MID 940/3200 / LARGE 1153/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 55.5 [53.8, 57.3] | 52.7 [50.8, 54.4] | SMALL 80.7 / MID 24.5 / LARGE 52.0 | SMALL 1084/1343 / MID 243/992 / LARGE 449/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 57.1 [53.7, 61.5] | 48.6 [44.2, 54.7] | SMALL 90.6 / MID 10.0 / LARGE 50.0 | SMALL 29/32 / MID 2/20 / LARGE 9/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 37.8 | 32.9 | SMALL 60.8 / MID 33.2 / LARGE 10.3 | SMALL 4027/6620 / MID 2610/7855 / LARGE 438/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 37.1 | 31.6 | SMALL 62.8 / MID 30.5 / LARGE 8.5 | SMALL 2177/3467 / MID 1234/4045 / LARGE 186/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 45.6 [42.7, 48.7] | 32.6 [29.8, 35.5] | SMALL 79.2 / MID 22.2 / LARGE 5.4 | SMALL 320/404 / MID 65/293 / LARGE 9/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 37.9 [32.1, 46.2] | 35.0 [26.2, 43.7] | SMALL 87.5 / MID 16.7 / LARGE 22.2 | SMALL 7/8 / MID 2/12 / LARGE 2/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 61.7 | 38.1 | SMALL 62.8 / MID 50.0 / LARGE 47.5 | SMALL 801/1275 / MID 25/50 / LARGE 29/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 68.2 | 43.7 | SMALL 68.8 / MID 54.5 / LARGE 70.6 | SMALL 422/613 / MID 18/33 / LARGE 12/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 88.9 [77.8, 97.2] | 75.6 [68.4, 99.2] | SMALL 87.5 / MID 100.0 / LARGE 100.0 | SMALL 28/32 / MID 1/1 / LARGE 3/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 100.0 [83.9, 100.0] | 100.0 [79.6, 100.0] | SMALL 100.0 / MID 100.0 / LARGE 100.0 | SMALL 10/10 / MID 1/1 / LARGE 3/3 | 33.3 / 71.4 |
| full+mask-V1 | all | window | 45763 | 48.9 | 36.5 | SMALL 91.4 / MID 19.4 / LARGE 10.6 | SMALL 18319/20036 / MID 2945/15159 / LARGE 1125/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 50.4 | 36.5 | SMALL 93.9 / MID 17.4 / LARGE 10.7 | SMALL 9305/9913 / MID 1264/7278 / LARGE 513/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 48.2 [46.7, 49.9] | 33.2 [31.6, 34.9] | SMALL 96.8 / MID 12.9 / LARGE 8.6 | SMALL 1706/1762 / MID 164/1274 / LARGE 88/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 45.1 [43.4, 48.6] | 22.5 [22.0, 29.4] | SMALL 100.0 / MID 3.0 / LARGE 0.0 | SMALL 50/50 / MID 1/33 / LARGE 0/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 51.3 | 37.3 | SMALL 91.8 / MID 17.3 / LARGE 12.4 | SMALL 11147/12141 / MID 1252/7254 / LARGE 775/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 54.2 | 37.5 | SMALL 94.7 / MID 14.0 / LARGE 12.8 | SMALL 5522/5833 / MID 449/3200 / LARGE 331/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 47.5 [45.9, 49.0] | 33.9 [32.1, 35.5] | SMALL 96.5 / MID 14.0 / LARGE 9.6 | SMALL 1296/1343 / MID 139/992 / LARGE 83/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 47.1 [44.4, 50.7] | 24.3 [23.3, 31.1] | SMALL 100.0 / MID 5.0 / LARGE 0.0 | SMALL 32/32 / MID 1/20 / LARGE 0/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 42.8 | 33.5 | SMALL 90.3 / MID 21.3 / LARGE 8.3 | SMALL 5978/6620 / MID 1674/7855 / LARGE 350/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 43.2 | 33.4 | SMALL 92.2 / MID 19.9 / LARGE 8.3 | SMALL 3198/3467 / MID 806/4045 / LARGE 182/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 50.0 [46.5, 53.1] | 30.6 [27.7, 33.6] | SMALL 97.8 / MID 9.6 / LARGE 5.4 | SMALL 395/404 / MID 28/293 / LARGE 9/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 27.6 [27.6, 35.7] | 14.4 [14.4, 27.9] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 8/8 / MID 0/12 / LARGE 0/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 87.5 | 39.1 | SMALL 93.6 / MID 38.0 / LARGE 0.0 | SMALL 1194/1275 / MID 19/50 / LARGE 0/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 89.6 | 39.8 | SMALL 95.4 / MID 27.3 / LARGE 0.0 | SMALL 585/613 / MID 9/33 / LARGE 0/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 88.9 [77.8, 97.2] | 31.4 [29.2, 49.3] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 32/32 / MID 0/1 / LARGE 0/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 71.4 [63.6, 90.9] | 27.8 [25.9, 47.5] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 10/10 / MID 0/1 / LARGE 0/3 | 33.3 / 71.4 |
| full+mask-V2 | all | window | 45763 | 42.5 | 28.9 | SMALL 85.5 / MID 8.2 / LARGE 10.0 | SMALL 17131/20036 / MID 1243/15159 / LARGE 1060/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 43.9 | 27.6 | SMALL 89.4 / MID 5.4 / LARGE 8.4 | SMALL 8858/9913 / MID 394/7278 / LARGE 405/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 42.9 [41.2, 44.3] | 25.4 [24.0, 26.4] | SMALL 92.6 / MID 3.0 / LARGE 6.9 | SMALL 1632/1762 / MID 38/1274 / LARGE 71/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 44.2 [42.5, 46.6] | 20.6 [19.9, 23.0] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 50/50 / MID 0/33 / LARGE 0/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 44.6 | 29.4 | SMALL 84.5 / MID 8.5 / LARGE 9.4 | SMALL 10257/12141 / MID 615/7254 / LARGE 587/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 48.1 | 29.1 | SMALL 88.9 / MID 5.9 / LARGE 8.3 | SMALL 5184/5833 / MID 188/3200 / LARGE 216/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 42.3 [40.8, 44.2] | 26.0 [24.6, 27.3] | SMALL 93.1 / MID 3.6 / LARGE 7.6 | SMALL 1251/1343 / MID 36/992 / LARGE 66/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 45.7 [43.3, 48.5] | 20.9 [20.1, 24.9] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 32/32 / MID 0/20 / LARGE 0/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 36.7 | 27.1 | SMALL 87.3 / MID 7.9 / LARGE 11.1 | SMALL 5778/6620 / MID 623/7855 / LARGE 472/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 36.4 | 24.8 | SMALL 90.6 / MID 5.0 / LARGE 8.6 | SMALL 3141/3467 / MID 203/4045 / LARGE 189/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 43.3 [40.2, 46.6] | 22.5 [20.6, 24.4] | SMALL 90.6 / MID 1.0 / LARGE 3.0 | SMALL 366/404 / MID 3/293 / LARGE 5/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 27.6 [27.6, 30.8] | 14.4 [14.4, 15.7] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 8/8 / MID 0/12 / LARGE 0/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 79.5 | 32.6 | SMALL 86.0 / MID 10.0 / LARGE 1.6 | SMALL 1096/1275 / MID 5/50 / LARGE 1/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 80.8 | 33.3 | SMALL 86.9 / MID 9.1 / LARGE 0.0 | SMALL 533/613 / MID 3/33 / LARGE 0/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 83.3 [68.0, 94.4] | 30.8 [28.0, 49.3] | SMALL 93.8 / MID 0.0 / LARGE 0.0 | SMALL 30/32 / MID 0/1 / LARGE 0/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 71.4 [63.6, 90.9] | 29.0 [27.5, 47.6] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 10/10 / MID 0/1 / LARGE 0/3 | 33.3 / 71.4 |
| full+mask-V3 | all | window | 45763 | 43.4 | 24.8 | SMALL 93.5 / MID 5.6 / LARGE 2.6 | SMALL 18734/20036 / MID 854/15159 / LARGE 280/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 44.4 | 23.5 | SMALL 95.1 / MID 4.0 / LARGE 1.1 | SMALL 9428/9913 / MID 290/7278 / LARGE 55/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 43.5 [42.1, 45.0] | 21.5 [20.9, 22.1] | SMALL 98.8 / MID 2.0 / LARGE 0.1 | SMALL 1741/1762 / MID 26/1274 / LARGE 1/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 45.1 [42.9, 46.7] | 22.6 [20.1, 23.2] | SMALL 100.0 / MID 3.0 / LARGE 0.0 | SMALL 50/50 / MID 1/33 / LARGE 0/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 46.8 | 26.7 | SMALL 93.4 / MID 6.4 / LARGE 3.3 | SMALL 11339/12141 / MID 465/7254 / LARGE 208/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 49.2 | 25.5 | SMALL 95.0 / MID 4.6 / LARGE 1.4 | SMALL 5543/5833 / MID 146/3200 / LARGE 37/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 42.3 [40.7, 43.9] | 21.3 [20.4, 22.3] | SMALL 98.8 / MID 2.5 / LARGE 0.0 | SMALL 1327/1343 / MID 25/992 / LARGE 0/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 47.1 [43.9, 49.2] | 24.3 [20.4, 25.1] | SMALL 100.0 / MID 5.0 / LARGE 0.0 | SMALL 32/32 / MID 1/20 / LARGE 0/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 35.6 | 21.0 | SMALL 93.8 / MID 5.0 / LARGE 1.5 | SMALL 6211/6620 / MID 389/7855 / LARGE 64/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 35.8 | 20.0 | SMALL 95.4 / MID 3.6 / LARGE 0.8 | SMALL 3307/3467 / MID 144/4045 / LARGE 18/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 46.4 [43.1, 49.4] | 21.7 [20.4, 22.9] | SMALL 98.8 / MID 0.3 / LARGE 0.6 | SMALL 399/404 / MID 1/293 / LARGE 1/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 27.6 [27.6, 30.8] | 14.4 [14.4, 15.7] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 8/8 / MID 0/12 / LARGE 0/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 86.0 | 37.1 | SMALL 92.9 / MID 0.0 / LARGE 13.1 | SMALL 1184/1275 / MID 0/50 / LARGE 8/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 87.2 | 31.1 | SMALL 94.3 / MID 0.0 / LARGE 0.0 | SMALL 578/613 / MID 0/33 / LARGE 0/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 88.9 [77.8, 97.2] | 31.8 [29.6, 49.3] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 32/32 / MID 0/1 / LARGE 0/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 71.4 [63.6, 90.9] | 29.0 [27.5, 47.6] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 10/10 / MID 0/1 / LARGE 0/3 | 33.3 / 71.4 |
| full+mask-yaw | all | window | 45763 | 35.6 | 34.8 | SMALL 27.1 / MID 22.8 / LARGE 70.1 | SMALL 5427/20036 / MID 3463/15159 / LARGE 7410/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 34.8 | 33.9 | SMALL 27.4 / MID 20.5 / LARGE 71.8 | SMALL 2721/9913 / MID 1492/7278 / LARGE 3443/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 39.8 [38.2, 41.4] | 37.5 [35.9, 39.0] | SMALL 25.7 / MID 20.4 / LARGE 88.3 | SMALL 452/1762 / MID 260/1274 / LARGE 905/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 38.1 [34.0, 41.0] | 33.3 [28.8, 37.5] | SMALL 14.0 / MID 18.2 / LARGE 100.0 | SMALL 7/50 / MID 6/33 / LARGE 30/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 38.3 | 37.1 | SMALL 27.8 / MID 25.0 / LARGE 74.0 | SMALL 3376/12141 / MID 1816/7254 / LARGE 4642/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 37.7 | 36.4 | SMALL 28.5 / MID 23.4 / LARGE 75.9 | SMALL 1664/5833 / MID 748/3200 / LARGE 1970/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 42.1 [40.4, 43.8] | 39.9 [38.2, 41.6] | SMALL 26.0 / MID 24.1 / LARGE 87.8 | SMALL 349/1343 / MID 239/992 / LARGE 759/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 38.6 [35.5, 43.1] | 35.6 [31.2, 41.4] | SMALL 12.5 / MID 25.0 / LARGE 100.0 | SMALL 4/32 / MID 5/20 / LARGE 18/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 31.8 | 31.3 | SMALL 23.9 / MID 20.8 / LARGE 64.4 | SMALL 1585/6620 / MID 1634/7855 / LARGE 2726/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 31.2 | 30.5 | SMALL 24.0 / MID 18.1 / LARGE 66.7 | SMALL 833/3467 / MID 734/4045 / LARGE 1458/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 31.5 [28.2, 34.8] | 28.7 [25.9, 31.7] | SMALL 24.3 / MID 8.2 / LARGE 89.8 | SMALL 98/404 / MID 24/293 / LARGE 150/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 34.5 [27.8, 39.3] | 22.3 [16.1, 30.6] | SMALL 0.0 / MID 8.3 / LARGE 100.0 | SMALL 0/8 / MID 1/12 / LARGE 9/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 37.6 | 24.5 | SMALL 36.5 / MID 26.0 / LARGE 68.9 | SMALL 466/1275 / MID 13/50 / LARGE 42/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 37.6 | 25.7 | SMALL 36.5 / MID 30.3 / LARGE 88.2 | SMALL 224/613 / MID 10/33 / LARGE 15/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 30.6 [15.2, 45.9] | 20.5 [11.1, 42.8] | SMALL 25.0 / MID 0.0 / LARGE 100.0 | SMALL 8/32 / MID 0/1 / LARGE 3/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 42.9 [20.0, 52.0] | 29.7 [14.5, 52.8] | SMALL 30.0 / MID 0.0 / LARGE 100.0 | SMALL 3/10 / MID 0/1 / LARGE 3/3 | 33.3 / 71.4 |
| full+mask-lead | all | window | 45763 | 49.3 | 42.1 | SMALL 81.3 / MID 27.2 / LARGE 20.2 | SMALL 16286/20036 / MID 4130/15159 / LARGE 2140/10568 | 33.3 / 43.8 |
|  | all | drive | 21987 | 50.9 | 42.7 | SMALL 84.8 / MID 24.0 / LARGE 21.6 | SMALL 8408/9913 / MID 1749/7278 / LARGE 1034/4796 | 33.3 / 45.1 |
|  | all | vehicle | 4061 | 53.8 [52.4, 55.3] | 45.5 [43.9, 47.1] | SMALL 93.2 / MID 23.5 / LARGE 23.8 | SMALL 1642/1762 / MID 299/1274 / LARGE 244/1025 | 33.3 / 43.4 |
|  | all | model | 113 | 55.8 [53.0, 58.4] | 44.7 [40.3, 48.9] | SMALL 100.0 / MID 21.2 / LARGE 20.0 | SMALL 50/50 / MID 7/33 / LARGE 6/30 | 33.3 / 44.2 |
|  | seen | window | 25667 | 53.7 | 45.7 | SMALL 84.4 / MID 27.2 / LARGE 25.2 | SMALL 10241/12141 / MID 1970/7254 / LARGE 1580/6272 | 33.3 / 47.3 |
|  | seen | drive | 11627 | 57.0 | 46.7 | SMALL 89.4 / MID 22.2 / LARGE 27.3 | SMALL 5213/5833 / MID 710/3200 / LARGE 709/2594 | 33.3 / 50.2 |
|  | seen | vehicle | 3199 | 53.7 [52.1, 55.5] | 46.6 [44.7, 48.4] | SMALL 92.8 / MID 24.7 / LARGE 26.2 | SMALL 1246/1343 / MID 245/992 / LARGE 226/864 | 33.3 / 42.0 |
|  | seen | model | 70 | 58.6 [53.1, 60.9] | 48.0 [40.7, 51.6] | SMALL 100.0 / MID 20.0 / LARGE 27.8 | SMALL 32/32 / MID 4/20 / LARGE 5/18 | 33.3 / 45.7 |
|  | held-out | window | 18710 | 40.6 | 35.1 | SMALL 73.9 / MID 27.3 / LARGE 13.2 | SMALL 4894/6620 / MID 2144/7855 / LARGE 557/4235 | 33.3 / 42.0 |
|  | held-out | drive | 9697 | 41.1 | 35.8 | SMALL 75.7 / MID 25.5 / LARGE 14.9 | SMALL 2625/3467 / MID 1033/4045 / LARGE 325/2185 | 33.3 / 41.7 |
|  | held-out | vehicle | 864 | 52.8 [49.8, 56.2] | 39.4 [35.6, 43.0] | SMALL 93.3 / MID 19.5 / LARGE 13.2 | SMALL 377/404 / MID 57/293 / LARGE 22/167 | 33.3 / 46.8 |
|  | held-out | model | 29 | 41.4 [37.9, 48.1] | 36.2 [31.9, 45.4] | SMALL 100.0 / MID 25.0 / LARGE 11.1 | SMALL 8/8 / MID 3/12 / LARGE 1/9 | 33.3 / 41.4 |
|  | absent | window | 1386 | 84.4 | 38.9 | SMALL 90.3 / MID 32.0 / LARGE 4.9 | SMALL 1151/1275 / MID 16/50 / LARGE 3/61 | 33.3 / 92.0 |
|  | absent | drive | 663 | 86.9 | 36.0 | SMALL 93.0 / MID 18.2 / LARGE 0.0 | SMALL 570/613 / MID 6/33 / LARGE 0/17 | 33.3 / 92.5 |
|  | absent | vehicle | 36 | 88.9 [77.8, 97.2] | 31.4 [29.2, 49.3] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 32/32 / MID 0/1 / LARGE 0/3 | 33.3 / 88.9 |
|  | absent | model | 14 | 71.4 [63.6, 90.9] | 27.8 [25.9, 47.5] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | SMALL 10/10 / MID 0/1 / LARGE 0/3 | 33.3 / 71.4 |

bev_ice (BEV / ICE):

| run | stratum | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | units right per class | chance uniform / majority % |
|---|---|---|---|---|---|---|---|---|
| full | all | window | 29585 | 81.8 | 74.7 | BEV 78.7 / ICE 82.5 | BEV 4268/5420 / ICE 19937/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 86.6 | 79.4 | BEV 80.9 / ICE 87.8 | BEV 1969/2435 / ICE 10464/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 87.8 [86.6, 89.1] | 84.7 [83.3, 86.3] | BEV 86.6 / ICE 88.2 | BEV 538/621 / ICE 1667/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 97.7 [96.2, 100.0] | 96.0 [93.4, 100.0] | BEV 93.3 / ICE 98.6 | BEV 14/15 / ICE 72/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 82.2 | 75.9 | BEV 81.5 / ICE 82.4 | BEV 2433/2985 / ICE 10424/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 86.7 | 79.5 | BEV 83.9 / ICE 87.3 | BEV 960/1144 / ICE 5139/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 86.1 [84.7, 87.6] | 83.5 [81.8, 85.1] | BEV 86.4 / ICE 86.0 | BEV 471/545 / ICE 1287/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 98.0 [95.8, 100.0] | 96.5 [93.2, 100.0] | BEV 88.9 / ICE 100.0 | BEV 8/9 / ICE 42/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 81.2 | 73.9 | BEV 75.4 / ICE 82.6 | BEV 1835/2435 / ICE 8784/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 86.4 | 79.9 | BEV 78.2 / ICE 88.3 | BEV 1009/1291 / ICE 4951/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 94.2 [92.0, 96.2] | 90.0 [86.2, 93.6] | BEV 86.2 / ICE 95.8 | BEV 69/80 / ICE 367/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 100.0 [96.0, 100.0] | 100.0 [93.3, 100.0] | BEV 100.0 / ICE 100.0 | BEV 6/6 / ICE 21/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 83.3 | - | BEV - / ICE 83.3 | BEV 0/0 / ICE 729/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 87.2 | - | BEV - / ICE 87.2 | BEV 0/0 / ICE 374/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 95.5 [86.4, 100.0] | - | BEV - / ICE 95.5 | BEV 0/0 / ICE 21/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 90.0 [83.3, 100.0] | - | BEV - / ICE 90.0 | BEV 0/0 / ICE 9/10 | 50.0 / 100.0 |
| V1 | all | window | 29585 | 63.1 | 58.8 | BEV 83.9 / ICE 58.4 | BEV 4548/5420 / ICE 14113/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 65.9 | 60.3 | BEV 83.7 / ICE 62.2 | BEV 2039/2435 / ICE 7418/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 76.4 [75.0, 78.2] | 74.0 [72.3, 75.9] | BEV 92.9 / ICE 71.0 | BEV 577/621 / ICE 1343/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 72.7 [69.9, 77.8] | 67.9 [65.4, 73.4] | BEV 100.0 / ICE 67.1 | BEV 15/15 / ICE 49/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 68.6 | 64.5 | BEV 90.3 / ICE 63.5 | BEV 2694/2985 / ICE 8038/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 72.2 | 66.2 | BEV 92.6 / ICE 68.3 | BEV 1059/1144 / ICE 4018/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 79.0 [77.2, 80.6] | 77.1 [75.0, 78.9] | BEV 94.3 / ICE 73.4 | BEV 514/545 / ICE 1098/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 80.4 [77.6, 87.0] | 75.4 [72.6, 82.3] | BEV 100.0 / ICE 76.2 | BEV 9/9 / ICE 32/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 56.7 | 53.0 | BEV 76.1 / ICE 52.3 | BEV 1854/2435 / ICE 5562/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 59.6 | 55.2 | BEV 75.9 / ICE 55.8 | BEV 980/1291 / ICE 3129/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 66.3 [62.1, 70.0] | 60.8 [56.1, 65.0] | BEV 83.8 / ICE 62.7 | BEV 67/80 / ICE 240/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 63.0 [53.8, 70.4] | 61.6 [53.6, 68.2] | BEV 100.0 / ICE 52.4 | BEV 6/6 / ICE 11/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 58.6 | - | BEV - / ICE 58.6 | BEV 0/0 / ICE 513/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 63.2 | - | BEV - / ICE 63.2 | BEV 0/0 / ICE 271/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 59.1 [36.4, 77.3] | - | BEV - / ICE 59.1 | BEV 0/0 / ICE 13/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 60.0 [37.5, 83.3] | - | BEV - / ICE 60.0 | BEV 0/0 / ICE 6/10 | 50.0 / 100.0 |
| V2 | all | window | 29585 | 74.7 | 66.1 | BEV 66.5 / ICE 76.6 | BEV 3603/5420 / ICE 18506/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 79.2 | 69.7 | BEV 68.5 / ICE 81.4 | BEV 1669/2435 / ICE 9704/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 84.6 [83.3, 86.1] | 80.6 [78.9, 82.3] | BEV 78.7 / ICE 86.6 | BEV 489/621 / ICE 1637/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 97.7 [95.1, 98.8] | 96.0 [92.2, 97.9] | BEV 93.3 / ICE 98.6 | BEV 14/15 / ICE 72/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 75.5 | 68.1 | BEV 71.7 / ICE 76.4 | BEV 2140/2985 / ICE 9670/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 80.6 | 72.2 | BEV 78.3 / ICE 81.1 | BEV 896/1144 / ICE 4774/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 83.2 [81.7, 85.2] | 79.9 [78.2, 82.2] | BEV 79.6 / ICE 84.6 | BEV 434/545 / ICE 1265/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 98.0 [95.6, 100.0] | 96.8 [93.1, 100.0] | BEV 100.0 / ICE 97.6 | BEV 9/9 / ICE 41/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 73.3 | 64.0 | BEV 60.1 / ICE 76.4 | BEV 1463/2435 / ICE 8121/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 77.4 | 67.6 | BEV 59.9 / ICE 81.4 | BEV 773/1291 / ICE 4563/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 89.2 [86.2, 91.8] | 81.3 [76.8, 85.5] | BEV 70.0 / ICE 93.2 | BEV 56/80 / ICE 357/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 96.3 [92.3, 96.3] | 94.3 [89.2, 94.3] | BEV 83.3 / ICE 100.0 | BEV 5/6 / ICE 21/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 81.7 | - | BEV - / ICE 81.7 | BEV 0/0 / ICE 715/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 85.5 | - | BEV - / ICE 85.5 | BEV 0/0 / ICE 367/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 22/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 10/10 | 50.0 / 100.0 |
| V3 | all | window | 29585 | 81.2 | 72.7 | BEV 69.3 / ICE 83.9 | BEV 3754/5420 / ICE 20279/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 85.1 | 76.2 | BEV 70.3 / ICE 88.1 | BEV 1712/2435 / ICE 10506/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 92.6 [91.5, 93.6] | 90.1 [88.8, 91.4] | BEV 86.3 / ICE 94.7 | BEV 536/621 / ICE 1790/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 98.9 [97.4, 98.8] | 97.9 [95.3, 97.9] | BEV 93.3 / ICE 100.0 | BEV 14/15 / ICE 73/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 83.4 | 76.6 | BEV 77.2 / ICE 84.9 | BEV 2304/2985 / ICE 10744/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 87.5 | 80.2 | BEV 82.0 / ICE 88.6 | BEV 938/1144 / ICE 5216/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 92.4 [91.4, 93.5] | 90.5 [89.1, 91.9] | BEV 88.8 / ICE 93.7 | BEV 484/545 / ICE 1402/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 100.0 [98.0, 100.0] | 100.0 [96.7, 100.0] | BEV 100.0 / ICE 100.0 | BEV 9/9 / ICE 42/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 78.3 | 68.3 | BEV 59.5 / ICE 82.6 | BEV 1450/2435 / ICE 8786/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 82.4 | 72.5 | BEV 60.0 / ICE 87.6 | BEV 774/1291 / ICE 4908/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 92.9 [90.8, 95.1] | 86.2 [82.1, 90.3] | BEV 67.5 / ICE 98.2 | BEV 54/80 / ICE 376/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 96.3 [92.3, 96.3] | 94.3 [85.2, 94.3] | BEV 83.3 / ICE 100.0 | BEV 5/6 / ICE 21/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 85.6 | - | BEV - / ICE 85.6 | BEV 0/0 / ICE 749/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 89.0 | - | BEV - / ICE 89.0 | BEV 0/0 / ICE 382/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 22/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 10/10 | 50.0 / 100.0 |
| full+mask-V1 | all | window | 29585 | 77.0 | 61.9 | BEV 38.5 / ICE 85.6 | BEV 2087/5420 / ICE 20691/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 78.6 | 61.5 | BEV 35.1 / ICE 87.5 | BEV 855/2435 / ICE 10435/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 82.4 [81.0, 83.8] | 71.6 [69.3, 73.8] | BEV 41.9 / ICE 95.7 | BEV 260/621 / ICE 1810/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 86.4 [84.3, 89.8] | 62.9 [57.4, 75.4] | BEV 20.0 / ICE 100.0 | BEV 3/15 / ICE 73/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 79.4 | 66.4 | BEV 44.8 / ICE 87.6 | BEV 1337/2985 / ICE 11088/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 82.4 | 66.9 | BEV 42.9 / ICE 90.1 | BEV 491/1144 / ICE 5304/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 81.9 [80.3, 83.4] | 72.8 [70.7, 75.0] | BEV 45.3 / ICE 95.2 | BEV 247/545 / ICE 1424/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 88.2 [85.1, 93.4] | 71.7 [64.0, 86.5] | BEV 33.3 / ICE 100.0 | BEV 3/9 / ICE 42/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 73.4 | 56.9 | BEV 30.8 / ICE 83.2 | BEV 750/2435 / ICE 8844/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 74.0 | 56.5 | BEV 28.2 / ICE 84.6 | BEV 364/1291 / ICE 4739/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 83.6 [80.3, 87.0] | 58.8 [52.4, 64.0] | BEV 17.5 / ICE 97.4 | BEV 14/80 / ICE 373/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 77.8 [76.0, 82.5] | 43.8 [43.2, 60.5] | BEV 0.0 / ICE 100.0 | BEV 0/6 / ICE 21/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 86.7 | - | BEV - / ICE 86.7 | BEV 0/0 / ICE 759/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 91.4 | - | BEV - / ICE 91.4 | BEV 0/0 / ICE 392/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 22/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 10/10 | 50.0 / 100.0 |
| full+mask-V2 | all | window | 29585 | 75.3 | 59.4 | BEV 34.6 / ICE 84.4 | BEV 1877/5420 / ICE 20394/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 78.6 | 60.4 | BEV 31.8 / ICE 88.2 | BEV 774/2435 / ICE 10516/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 77.5 [75.9, 79.0] | 62.6 [60.3, 64.8] | BEV 29.1 / ICE 93.4 | BEV 181/621 / ICE 1766/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 81.8 [80.0, 84.1] | 45.0 [44.4, 52.3] | BEV 0.0 / ICE 98.6 | BEV 0/15 / ICE 72/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 74.9 | 60.2 | BEV 36.8 / ICE 83.9 | BEV 1099/2985 / ICE 10622/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 79.0 | 61.4 | BEV 35.3 / ICE 87.5 | BEV 404/1144 / ICE 5149/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 76.1 [74.3, 77.8] | 63.2 [60.6, 65.4] | BEV 31.6 / ICE 92.4 | BEV 172/545 / ICE 1382/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 82.4 [80.4, 84.2] | 45.2 [44.6, 55.6] | BEV 0.0 / ICE 100.0 | BEV 0/9 / ICE 42/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 74.8 | 58.3 | BEV 32.0 / ICE 84.6 | BEV 778/2435 / ICE 8997/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 77.5 | 59.4 | BEV 28.7 / ICE 88.7 | BEV 370/1291 / ICE 4971/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 82.7 [79.0, 86.2] | 55.2 [50.3, 61.3] | BEV 12.5 / ICE 97.4 | BEV 10/80 / ICE 373/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 77.8 [76.0, 81.5] | 43.8 [43.2, 59.0] | BEV 0.0 / ICE 100.0 | BEV 0/6 / ICE 21/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 88.6 | - | BEV - / ICE 88.6 | BEV 0/0 / ICE 775/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 92.3 | - | BEV - / ICE 92.3 | BEV 0/0 / ICE 396/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 95.5 [86.4, 100.0] | - | BEV - / ICE 95.5 | BEV 0/0 / ICE 21/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 90.0 [83.3, 100.0] | - | BEV - / ICE 90.0 | BEV 0/0 / ICE 9/10 | 50.0 / 100.0 |
| full+mask-V3 | all | window | 29585 | 81.2 | 63.0 | BEV 30.3 / ICE 92.6 | BEV 1641/5420 / ICE 22381/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 83.7 | 63.4 | BEV 27.2 / ICE 95.2 | BEV 663/2435 / ICE 11345/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 81.5 [80.1, 82.9] | 66.6 [64.3, 69.0] | BEV 29.8 / ICE 98.5 | BEV 185/621 / ICE 1862/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 81.8 [80.0, 83.5] | 45.0 [44.4, 51.5] | BEV 0.0 / ICE 98.6 | BEV 0/15 / ICE 72/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 81.2 | 64.6 | BEV 33.4 / ICE 92.5 | BEV 996/2985 / ICE 11708/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 85.3 | 66.6 | BEV 32.3 / ICE 95.6 | BEV 369/1144 / ICE 5629/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 80.6 [79.1, 82.3] | 67.7 [65.3, 70.1] | BEV 32.5 / ICE 98.2 | BEV 177/545 / ICE 1469/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 82.4 [80.4, 85.1] | 45.2 [44.6, 57.0] | BEV 0.0 / ICE 100.0 | BEV 0/9 / ICE 42/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 80.3 | 60.9 | BEV 26.5 / ICE 92.6 | BEV 645/2435 / ICE 9850/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 81.3 | 60.2 | BEV 22.8 / ICE 94.8 | BEV 294/1291 / ICE 5310/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 84.4 [80.6, 88.0] | 55.7 [49.5, 61.9] | BEV 11.2 / ICE 99.7 | BEV 9/80 / ICE 382/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 77.8 [76.0, 80.8] | 43.8 [43.2, 44.7] | BEV 0.0 / ICE 100.0 | BEV 0/6 / ICE 21/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 94.1 | - | BEV - / ICE 94.1 | BEV 0/0 / ICE 823/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 94.6 | - | BEV - / ICE 94.6 | BEV 0/0 / ICE 406/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 95.5 [86.4, 100.0] | - | BEV - / ICE 95.5 | BEV 0/0 / ICE 21/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 90.0 [83.3, 100.0] | - | BEV - / ICE 90.0 | BEV 0/0 / ICE 9/10 | 50.0 / 100.0 |
| full+mask-yaw | all | window | 29585 | 82.4 | 75.2 | BEV 77.9 / ICE 83.4 | BEV 4221/5420 / ICE 20152/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 86.7 | 79.7 | BEV 82.7 / ICE 87.5 | BEV 2013/2435 / ICE 10431/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 86.2 [84.9, 87.5] | 83.1 [81.4, 84.7] | BEV 87.4 / ICE 85.8 | BEV 543/621 / ICE 1623/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 98.9 [97.5, 100.0] | 97.9 [95.7, 100.0] | BEV 93.3 / ICE 100.0 | BEV 14/15 / ICE 73/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 81.9 | 75.3 | BEV 79.7 / ICE 82.4 | BEV 2379/2985 / ICE 10426/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 85.5 | 78.3 | BEV 85.0 / ICE 85.7 | BEV 972/1144 / ICE 5043/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 84.1 [82.6, 85.7] | 81.5 [79.7, 83.3] | BEV 86.6 / ICE 83.2 | BEV 472/545 / ICE 1245/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 98.0 [95.8, 100.0] | 96.5 [93.2, 100.0] | BEV 88.9 / ICE 100.0 | BEV 8/9 / ICE 42/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 82.9 | 75.6 | BEV 75.6 / ICE 84.6 | BEV 1842/2435 / ICE 8998/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 87.8 | 81.7 | BEV 80.6 / ICE 89.4 | BEV 1041/1291 / ICE 5012/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 94.6 [92.4, 96.7] | 91.1 [87.5, 94.2] | BEV 92.5 / ICE 95.0 | BEV 74/80 / ICE 364/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 100.0 [96.2, 100.0] | 100.0 [94.2, 100.0] | BEV 100.0 / ICE 100.0 | BEV 6/6 / ICE 21/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 83.2 | - | BEV - / ICE 83.2 | BEV 0/0 / ICE 728/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 87.6 | - | BEV - / ICE 87.6 | BEV 0/0 / ICE 376/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 22/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 10/10 | 50.0 / 100.0 |
| full+mask-lead | all | window | 29585 | 85.6 | 78.4 | BEV 75.5 / ICE 87.9 | BEV 4094/5420 / ICE 21242/24165 | 50.0 / 81.7 |
|  | all | drive | 14355 | 89.7 | 82.7 | BEV 77.5 / ICE 92.2 | BEV 1886/2435 / ICE 10985/11920 | 50.0 / 83.0 |
|  | all | vehicle | 2512 | 91.3 [90.2, 92.3] | 88.6 [87.2, 89.9] | BEV 85.3 / ICE 93.3 | BEV 530/621 / ICE 1764/1891 | 50.0 / 75.3 |
|  | all | model | 88 | 97.7 [96.3, 100.0] | 96.0 [93.7, 100.0] | BEV 93.3 / ICE 98.6 | BEV 14/15 / ICE 72/73 | 50.0 / 83.0 |
|  | seen | window | 15640 | 85.7 | 79.4 | BEV 79.1 / ICE 87.3 | BEV 2361/2985 / ICE 11048/12655 | 50.0 / 80.9 |
|  | seen | drive | 7031 | 89.8 | 82.9 | BEV 80.9 / ICE 91.5 | BEV 926/1144 / ICE 5387/5887 | 50.0 / 83.7 |
|  | seen | vehicle | 2041 | 90.1 [88.8, 91.2] | 87.6 [86.1, 89.1] | BEV 85.0 / ICE 91.9 | BEV 463/545 / ICE 1375/1496 | 50.0 / 73.3 |
|  | seen | model | 51 | 98.0 [97.8, 100.0] | 96.5 [96.4, 100.0] | BEV 88.9 / ICE 100.0 | BEV 8/9 / ICE 42/42 | 50.0 / 82.4 |
|  | held-out | window | 13070 | 85.5 | 77.7 | BEV 71.2 / ICE 88.7 | BEV 1733/2435 / ICE 9436/10635 | 50.0 / 81.4 |
|  | held-out | drive | 6895 | 89.5 | 83.0 | BEV 74.4 / ICE 93.0 | BEV 960/1291 / ICE 5210/5604 | 50.0 / 81.3 |
|  | held-out | vehicle | 463 | 96.3 [94.6, 98.1] | 93.4 [90.1, 96.4] | BEV 86.2 / ICE 98.4 | BEV 69/80 / ICE 377/383 | 50.0 / 82.7 |
|  | held-out | model | 27 | 100.0 [96.0, 100.0] | 100.0 [93.3, 100.0] | BEV 100.0 / ICE 100.0 | BEV 6/6 / ICE 21/21 | 50.0 / 77.8 |
|  | absent | window | 875 | 86.6 | - | BEV - / ICE 86.6 | BEV 0/0 / ICE 758/875 | 50.0 / 100.0 |
|  | absent | drive | 429 | 90.4 | - | BEV - / ICE 90.4 | BEV 0/0 / ICE 388/429 | 50.0 / 100.0 |
|  | absent | vehicle | 22 | 95.5 [86.4, 100.0] | - | BEV - / ICE 95.5 | BEV 0/0 / ICE 21/22 | 50.0 / 100.0 |
|  | absent | model | 10 | 90.0 [83.3, 100.0] | - | BEV - / ICE 90.0 | BEV 0/0 / ICE 9/10 | 50.0 / 100.0 |

### commaCarSegments (`ccs`): held-out models against the internal per-model result (full models)

Every held-out `split_platform` test model present in the set: the external vehicle-level accuracy (devices right / devices) and the model-level decision with P(true), next to the internal result on the lab's own test windows of that model (internal_per_model.md, column `full`: drive accuracy % / decision (P true)). A model may span several platforms (e.g. gasoline and hybrid ports); `-` = no label for that task or no internal row.

| held-out model | platforms | veh | win | size label | size: veh right (acc %) | size: model decision (P true) | size: internal drive acc / decision (P true) | pt label | pt: veh right (acc %) | pt: model decision (P true) | pt: internal drive acc / decision (P true) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Chevrolet Equinox | 1 | 5 | 243 | SMALL | 5/5 (100.0) | SMALL (0.64) OK | 50.0 / SMALL (0.37) | ICE | 5/5 (100.0) | ICE (0.80) OK | 100.0 / ICE (0.84) |
| Ford F-150 Lightning | 1 | 6 | 113 | LARGE | 5/6 (83.3) | LARGE (0.55) OK | 95.0 / LARGE (0.73) | BEV | 4/6 (66.7) | BEV (0.52) OK | 86.0 / BEV (0.63) |
| Ford Ranger | 1 | 3 | 82 | LARGE | 2/3 (66.7) | SMALL (0.38) MISS | 90.0 / LARGE (0.74) | ICE | 3/3 (100.0) | ICE (0.67) OK | 90.0 / ICE (0.70) |
| Genesis G80 | 1 | 7 | 304 | LARGE | 1/7 (14.3) | SMALL (0.26) MISS | 0.0 / MID (0.13) x | ICE | 7/7 (100.0) | ICE (0.73) OK | 100.0 / ICE (0.73) |
| Genesis G90 | 1 | 9 | 249 | LARGE | 0/9 (0.0) | SMALL (0.06) MISS | 0.0 / MID (0.13) x | ICE | 9/9 (100.0) | ICE (0.73) OK | 100.0 / ICE (0.74) |
| Genesis GV60 | 1 | 2 | 97 | MID | 1/2 (50.0) | MID (0.52) OK | 97.0 / MID (0.61) | BEV | 2/2 (100.0) | BEV (0.78) OK | 99.0 / BEV (0.83) |
| Genesis GV70 | 1 | 8 | 200 | MID | 1/8 (12.5) | SMALL (0.36) MISS | 67.0 / MID (0.45) | ICE | 8/8 (100.0) | ICE (0.79) OK | 100.0 / ICE (0.73) |
| Genesis GV80 | 1 | 3 | 146 | LARGE | 0/3 (0.0) | SMALL (0.06) MISS | 62.0 / LARGE (0.51) | ICE | 3/3 (100.0) | ICE (0.89) OK | 84.0 / ICE (0.73) |
| Hyundai Azera | 1 | 1 | 5 | MID | 0/1 (0.0) | SMALL (0.32) MISS | 80.0 / MID (0.50) | - | - | - | - |
| Hyundai Ioniq | 4 | 26 | 935 | SMALL | 26/26 (100.0) | SMALL (0.70) OK | 69.0 / SMALL (0.59) | BEV | 5/5 (100.0) | BEV (0.83) OK | 90.0 / BEV (0.79) |
| Hyundai Kona | 3 | 20 | 538 | SMALL | 18/20 (90.0) | SMALL (0.63) OK | 86.0 / SMALL (0.60) | BEV | 18/20 (90.0) | BEV (0.75) OK | 70.0 / ICE (0.67) |
| Hyundai Santa Cruz | 1 | 1 | 49 | LARGE | 0/1 (0.0) | SMALL (0.09) MISS | 83.0 / LARGE (0.45) | ICE | 1/1 (100.0) | ICE (0.84) OK | 100.0 / ICE (0.69) |
| Hyundai Santa Fe | 4 | 62 | 2355 | MID | 39/62 (62.9) | MID (0.46) OK | 66.0 / MID (0.43) | ICE | 43/46 (93.5) | ICE (0.73) OK | 95.0 / ICE (0.73) |
| Jeep Grand Cherokee | 2 | 36 | 1438 | SMALL | 26/36 (72.2) | SMALL (0.49) OK | 22.0 / LARGE (0.23) x | ICE | 36/36 (100.0) | ICE (0.85) OK | 100.0 / ICE (0.93) |
| Kia Carnival | 1 | 15 | 477 | LARGE | 0/15 (0.0) | SMALL (0.07) MISS | 77.0 / LARGE (0.50) | ICE | 15/15 (100.0) | ICE (0.80) OK | 96.0 / ICE (0.82) |
| Kia K5 | 1 | 3 | 99 | MID | 0/3 (0.0) | SMALL (0.26) MISS | 25.0 / SMALL (0.40) x | ICE | 3/3 (100.0) | ICE (0.77) OK | 100.0 / ICE (0.75) |
| Kia Optima | 1 | 5 | 163 | MID | 0/5 (0.0) | SMALL (0.30) MISS | 85.0 / MID (0.56) | ICE | 5/5 (100.0) | ICE (0.70) OK | 100.0 / ICE (0.86) |
| Kia Sorento | 3 | 17 | 545 | MID | 0/17 (0.0) | SMALL (0.26) MISS | 57.0 / SMALL (0.40) x | ICE | 9/9 (100.0) | ICE (0.83) OK | 50.0 / ICE (0.66) |
| Kia Stinger | 2 | 10 | 479 | MID | 10/10 (100.0) | MID (0.56) OK | 0.0 / SMALL (0.33) x | ICE | 10/10 (100.0) | ICE (0.70) OK | 100.0 / ICE (0.77) |
| Lexus RX | 2 | 124 | 1931 | MID | 7/124 (5.6) | SMALL (0.32) MISS | 17.0 / SMALL (0.32) x | ICE | 60/69 (87.0) | ICE (0.69) OK | 100.0 / ICE (0.67) |
| Nissan Altima | 1 | 4 | 193 | MID | 0/4 (0.0) | SMALL (0.32) MISS | 0.0 / SMALL (0.20) x | ICE | 4/4 (100.0) | ICE (0.83) OK | 83.0 / ICE (0.63) |
| Nissan Leaf | 2 | 28 | 916 | SMALL | 17/28 (60.7) | SMALL (0.54) OK | 61.0 / SMALL (0.56) | BEV | 22/28 (78.6) | BEV (0.63) OK | 64.0 / BEV (0.60) |
| Tesla Model Y | 1 | 21 | 538 | MID | 17/21 (81.0) | MID (0.59) OK | 66.0 / MID (0.45) | BEV | 20/21 (95.2) | BEV (0.84) OK | 100.0 / BEV (0.83) |
| Toyota Avalon | 3 | 37 | 1250 | MID | 1/37 (2.7) | SMALL (0.29) MISS | 50.0 / SMALL (0.44) x | ICE | 6/6 (100.0) | ICE (0.82) OK | 100.0 / ICE (0.92) |
| Toyota Corolla | 2 | 272 | 1952 | SMALL | 259/272 (95.2) | SMALL (0.65) OK | 95.0 / SMALL (0.72) | ICE | 39/39 (100.0) | ICE (0.81) OK | 94.0 / ICE (0.72) |
| Toyota Sienna | 2 | 94 | 1854 | LARGE | 15/94 (16.0) | SMALL (0.19) MISS | 45.0 / LARGE (0.42) | ICE | 58/62 (93.5) | ICE (0.75) OK | 100.0 / ICE (0.82) |
| Volkswagen Atlas | 1 | 32 | 961 | LARGE | 0/32 (0.0) | SMALL (0.09) MISS | 0.0 / SMALL (0.14) x | ICE | 32/32 (100.0) | ICE (0.86) OK | 100.0 / ICE (0.89) |
| Volkswagen Jetta | 1 | 12 | 342 | SMALL | 12/12 (100.0) | SMALL (0.72) OK | 82.0 / SMALL (0.65) | ICE | 12/12 (100.0) | ICE (0.88) OK | 100.0 / ICE (0.88) |
| Volkswagen Passat | 1 | 8 | 256 | SMALL | 8/8 (100.0) | SMALL (0.68) OK | 81.0 / SMALL (0.56) | - | - | - | - |

### commaCarSegments (`ccs`): seen and absent models (full models; seen platform, new device)

Per car model of the seen stratum (training / validation models of the lab split, here on devices the lab never saw) and of the absent stratum (models outside the lab inventory): model-level hits, the median per-model vehicle accuracy, then the ten seen models with the lowest P(true) and every absent model.

size (SMALL / MID / LARGE), seen models: 40 / 70 right at model level; median per-model vehicle accuracy 59.6 %; devices right 1743 / 3245 (53.7 %).

| model | label | platforms | veh | win | veh right (acc %) | P(true) | model decision |
|---|---|---|---|---|---|---|---|
| Volkswagen Transporter | LARGE | 1 | 2 | 74 | 0/2 (0.0) | 0.07 | SMALL MISS |
| Cadillac Escalade ESV | LARGE | 1 | 1 | 8 | 0/1 (0.0) | 0.08 | SMALL MISS |
| Honda Odyssey | LARGE | 1 | 96 | 381 | 2/96 (2.1) | 0.08 | SMALL MISS |
| Honda Pilot | LARGE | 1 | 77 | 396 | 2/77 (2.6) | 0.09 | SMALL MISS |
| Honda Ridgeline | LARGE | 1 | 56 | 392 | 2/56 (3.6) | 0.17 | SMALL MISS |
| Ford Explorer | LARGE | 1 | 44 | 374 | 5/44 (11.4) | 0.21 | SMALL MISS |
| Chrysler Pacifica | LARGE | 4 | 184 | 1475 | 27/184 (14.7) | 0.21 | SMALL MISS |
| Cadillac XT4 | MID | 1 | 1 | 1 | 0/1 (0.0) | 0.21 | SMALL MISS |
| Ford Maverick | LARGE | 1 | 32 | 382 | 2/32 (6.2) | 0.21 | SMALL MISS |
| Subaru Ascent | MID | 1 | 34 | 389 | 0/34 (0.0) | 0.21 | SMALL MISS |

size (SMALL / MID / LARGE), absent models: 10 / 14 right at model level; median per-model vehicle accuracy 100.0 %; devices right 32 / 36 (88.9 %).

| model | label | platforms | veh | win | veh right (acc %) | P(true) | model decision |
|---|---|---|---|---|---|---|---|
| Buick Regal Sportback | MID | 1 | 1 | 50 | 0/1 (0.0) | 0.32 | SMALL MISS |
| Cadillac Escalade | LARGE | 1 | 1 | 1 | 0/1 (0.0) | 0.19 | SMALL MISS |
| Dodge Durango | LARGE | 1 | 1 | 11 | 0/1 (0.0) | 0.22 | SMALL MISS |
| Hyundai Elantra GT / i30 | SMALL | 1 | 2 | 100 | 2/2 (100.0) | 0.69 | SMALL OK |
| Hyundai Ioniq Plug-in | SMALL | 1 | 2 | 100 | 2/2 (100.0) | 0.70 | SMALL OK |
| Hyundai Staria | LARGE | 1 | 1 | 49 | 0/1 (0.0) | 0.04 | SMALL MISS |
| Lexus CT 200h | SMALL | 1 | 3 | 65 | 3/3 (100.0) | 0.55 | SMALL OK |
| SEAT Ateca | SMALL | 1 | 4 | 152 | 4/4 (100.0) | 0.75 | SMALL OK |
| Skoda Kamiq | SMALL | 1 | 1 | 35 | 1/1 (100.0) | 0.86 | SMALL OK |
| Skoda Octavia | SMALL | 1 | 7 | 286 | 7/7 (100.0) | 0.77 | SMALL OK |
| Toyota Mirai | SMALL | 1 | 6 | 296 | 6/6 (100.0) | 0.66 | SMALL OK |
| Toyota Prius v | SMALL | 1 | 3 | 50 | 3/3 (100.0) | 0.63 | SMALL OK |
| Volkswagen T-Roc | SMALL | 1 | 3 | 147 | 3/3 (100.0) | 0.78 | SMALL OK |
| Volkswagen Touran | SMALL | 1 | 1 | 44 | 1/1 (100.0) | 0.70 | SMALL OK |

bev_ice (BEV / ICE), seen models: 50 / 51 right at model level; median per-model vehicle accuracy 96.4 %; devices right 1772 / 2059 (86.1 %).

| model | label | platforms | veh | win | veh right (acc %) | P(true) | model decision |
|---|---|---|---|---|---|---|---|
| Honda e | BEV | 1 | 1 | 2 | 0/1 (0.0) | 0.34 | ICE MISS |
| Ford Bronco Sport | ICE | 1 | 8 | 94 | 7/8 (87.5) | 0.62 | ICE OK |
| Toyota Highlander | ICE | 2 | 225 | 765 | 148/225 (65.8) | 0.64 | ICE OK |
| Cadillac Escalade ESV | ICE | 1 | 1 | 8 | 1/1 (100.0) | 0.67 | ICE OK |
| Toyota RAV4 | ICE | 3 | 307 | 1165 | 224/307 (73.0) | 0.68 | ICE OK |
| Ford Focus | ICE | 1 | 1 | 24 | 1/1 (100.0) | 0.69 | ICE OK |
| Volkswagen Transporter | ICE | 1 | 2 | 74 | 2/2 (100.0) | 0.72 | ICE OK |
| Lexus IS | ICE | 2 | 16 | 558 | 14/16 (87.5) | 0.72 | ICE OK |
| Nissan Rogue | ICE | 1 | 5 | 100 | 5/5 (100.0) | 0.73 | ICE OK |
| Chevrolet Bolt EUV | BEV | 1 | 160 | 380 | 118/160 (73.8) | 0.73 | BEV OK |

bev_ice (BEV / ICE), absent models: 9 / 10 right at model level; median per-model vehicle accuracy 100.0 %; devices right 21 / 22 (95.5 %).

| model | label | platforms | veh | win | veh right (acc %) | P(true) | model decision |
|---|---|---|---|---|---|---|---|
| Buick Regal Sportback | ICE | 1 | 1 | 50 | 1/1 (100.0) | 0.78 | ICE OK |
| Cadillac Escalade | ICE | 1 | 1 | 1 | 0/1 (0.0) | 0.07 | BEV MISS |
| Dodge Durango | ICE | 1 | 1 | 11 | 1/1 (100.0) | 0.84 | ICE OK |
| Hyundai Elantra GT / i30 | ICE | 1 | 2 | 100 | 2/2 (100.0) | 0.58 | ICE OK |
| Hyundai Staria | ICE | 1 | 1 | 49 | 1/1 (100.0) | 0.68 | ICE OK |
| SEAT Ateca | ICE | 1 | 4 | 152 | 4/4 (100.0) | 0.82 | ICE OK |
| Skoda Kamiq | ICE | 1 | 1 | 35 | 1/1 (100.0) | 0.77 | ICE OK |
| Skoda Octavia | ICE | 1 | 7 | 286 | 7/7 (100.0) | 0.81 | ICE OK |
| Volkswagen T-Roc | ICE | 1 | 3 | 147 | 3/3 (100.0) | 0.84 | ICE OK |
| Volkswagen Touran | ICE | 1 | 1 | 44 | 1/1 (100.0) | 0.77 | ICE OK |

### commaCarSegments (`ccs`): lead source and automation state per stratum (full models)

`lead` = the platform's lead source (radar tracks / stock-ACC object / none = vision-only); `state`: human = das_frac < 0.5, engaged = das_frac >= 0.5 (the driving automation system or stock cruise active for most of the minute). Vehicle accuracy inside a cell uses the vehicle's windows of that cell only.

| factor | value | stratum | size: win / veh | size win acc % | size veh acc % (right/n) | pt: win / veh | pt win acc % | pt veh acc % (right/n) |
|---|---|---|---|---|---|---|---|---|
| lead | radar | all | 15539 / 1760 | 43.4 | 47.6 (837/1760) | 8078 / 900 | 77.9 | 82.3 (741/900) |
| lead | radar | seen | 8044 / 1237 | 47.8 | 44.6 (552/1237) | 4712 / 721 | 76.7 | 79.6 (574/721) |
| lead | radar | held-out | 7085 / 531 | 36.5 | 53.7 (285/531) | 3366 / 185 | 79.7 | 92.4 (171/185) |
| lead | radar | absent | 410 / 12 | 77.6 | 100.0 (12/12) | 0 / 0 | - | - |
| lead | scc | all | 4740 / 267 | 54.0 | 53.6 (143/267) | 3352 / 207 | 73.8 | 86.5 (179/207) |
| lead | scc | seen | 1666 / 150 | 53.7 | 46.0 (69/150) | 1075 / 126 | 74.0 | 85.7 (108/126) |
| lead | scc | held-out | 2963 / 113 | 52.8 | 61.9 (70/113) | 2194 / 79 | 74.3 | 88.6 (70/79) |
| lead | scc | absent | 111 / 4 | 91.0 | 100.0 (4/4) | 83 / 2 | 56.6 | 50.0 (1/2) |
| lead | none | all | 25484 / 2526 | 52.6 | 57.9 (1462/2526) | 18155 / 1700 | 85.0 | 89.6 (1524/1700) |
| lead | none | seen | 15957 / 2103 | 56.1 | 59.2 (1245/2103) | 9853 / 1367 | 85.8 | 88.7 (1212/1367) |
| lead | none | held-out | 8662 / 409 | 43.5 | 49.1 (201/409) | 7510 / 317 | 84.0 | 93.1 (295/317) |
| lead | none | absent | 865 / 25 | 79.3 | 84.0 (21/25) | 792 / 22 | 86.1 | 95.5 (21/22) |
| state | human | all | 27759 / 3486 | 51.0 | 55.7 (1943/3486) | 18356 / 2150 | 92.2 | 93.8 (2017/2150) |
| state | human | seen | 15361 / 2694 | 55.3 | 56.1 (1510/2694) | 9673 / 1697 | 92.4 | 92.6 (1572/1697) |
| state | human | held-out | 11562 / 785 | 43.3 | 53.9 (423/785) | 8187 / 444 | 91.9 | 97.5 (433/444) |
| state | human | absent | 836 / 34 | 79.8 | 91.2 (31/34) | 496 / 21 | 94.8 | 100.0 (21/21) |
| state | engaged | all | 18004 / 2700 | 47.6 | 48.3 (1303/2700) | 11229 / 1620 | 64.8 | 70.9 (1149/1620) |
| state | engaged | seen | 10306 / 2019 | 50.5 | 49.2 (993/2019) | 5967 / 1235 | 65.7 | 70.9 (876/1235) |
| state | engaged | held-out | 7148 / 675 | 40.9 | 43.7 (295/675) | 4883 / 374 | 63.3 | 70.6 (264/374) |
| state | engaged | absent | 550 / 30 | 79.6 | 90.0 (27/30) | 379 / 19 | 68.3 | 78.9 (15/19) |

### commaCarSegments (`ccs`): evidence per vehicle (full models; vehicle-level accuracy by labelled windows per device)

How many minutes of a new device are needed: devices bucketed by their number of labelled windows (1, 2-4, 5-9, 10+); cells = devices right / devices (accuracy %).

| task | stratum | devices | 1 window | 2-4 windows | 5-9 windows | 10+ windows |
|---|---|---|---|---|---|---|
| size | all | 4061 | 308/513 (60.0) | 771/1371 (56.2) | 438/822 (53.3) | 681/1355 (50.3) |
| size | seen | 3199 | 250/424 (59.0) | 650/1225 (53.1) | 372/736 (50.5) | 452/814 (55.5) |
| size | held-out | 864 | 59/93 (63.4) | 136/170 (80.0) | 70/90 (77.8) | 200/511 (39.1) |
| size | absent | 36 | 2/3 (66.7) | 2/2 (100.0) | - | 28/31 (90.3) |
| bev_ice | all | 2512 | 277/346 (80.1) | 681/830 (82.0) | 433/491 (88.2) | 814/845 (96.3) |
| bev_ice | seen | 2041 | 245/308 (79.5) | 650/796 (81.7) | 417/475 (87.8) | 446/462 (96.5) |
| bev_ice | held-out | 463 | 34/40 (85.0) | 35/40 (87.5) | 19/20 (95.0) | 348/363 (95.9) |
| bev_ice | absent | 22 | 0/1 (0.0) | 1/1 (100.0) | - | 20/20 (100.0) |

### commaCarSegments (`ccs`): predicted-share confusion per stratum (size; full model and full model with the yaw group masked)

Predicted class shares of the windows per true class and stratum (the confusion matrix as shares) and the mean probabilities; the yaw-masked run shows how much of the decision rests on the steering-derived yaw / curvature / lateral-acceleration group.

| run | stratum | true class | veh | win | predicted share SMALL, MID, LARGE % | mean P(SMALL, MID, LARGE) |
|---|---|---|---|---|---|---|
| full | all | SMALL | 1781 | 20036 | 78.1, 19.9, 1.9 | 0.66, 0.29, 0.05 |
| full | all | MID | 1291 | 15159 | 65.1, 32.6, 2.3 | 0.56, 0.38, 0.06 |
| full | all | LARGE | 1049 | 10568 | 54.1, 25.8, 20.1 | 0.47, 0.31, 0.22 |
| full | seen | SMALL | 1347 | 12141 | 81.3, 17.4, 1.3 | 0.68, 0.28, 0.04 |
| full | seen | MID | 1002 | 7254 | 66.7, 31.2, 2.1 | 0.58, 0.37, 0.06 |
| full | seen | LARGE | 879 | 6272 | 50.7, 24.5, 24.8 | 0.45, 0.29, 0.26 |
| full | held-out | SMALL | 406 | 6620 | 71.0, 25.8, 3.2 | 0.61, 0.33, 0.06 |
| full | held-out | MID | 294 | 7855 | 63.5, 33.9, 2.6 | 0.55, 0.39, 0.06 |
| full | held-out | LARGE | 168 | 4235 | 58.9, 27.8, 13.2 | 0.51, 0.33, 0.16 |
| full | absent | SMALL | 32 | 1275 | 85.2, 13.8, 1.0 | 0.71, 0.24, 0.04 |
| full | absent | MID | 1 | 50 | 68.0, 32.0, 0.0 | 0.61, 0.32, 0.07 |
| full | absent | LARGE | 3 | 61 | 75.4, 19.7, 4.9 | 0.63, 0.29, 0.08 |
| full+mask-yaw | all | SMALL | 1781 | 20036 | 27.1, 19.0, 53.9 | 0.28, 0.25, 0.47 |
| full+mask-yaw | all | MID | 1291 | 15159 | 20.4, 22.8, 56.8 | 0.24, 0.27, 0.48 |
| full+mask-yaw | all | LARGE | 1049 | 10568 | 15.7, 14.2, 70.1 | 0.20, 0.21, 0.59 |
| full+mask-yaw | seen | SMALL | 1347 | 12141 | 27.8, 17.7, 54.5 | 0.29, 0.24, 0.47 |
| full+mask-yaw | seen | MID | 1002 | 7254 | 21.8, 25.0, 53.1 | 0.25, 0.28, 0.46 |
| full+mask-yaw | seen | LARGE | 879 | 6272 | 13.2, 12.8, 74.0 | 0.18, 0.20, 0.62 |
| full+mask-yaw | held-out | SMALL | 406 | 6620 | 23.9, 21.9, 54.1 | 0.26, 0.26, 0.47 |
| full+mask-yaw | held-out | MID | 294 | 7855 | 19.1, 20.8, 60.1 | 0.23, 0.27, 0.50 |
| full+mask-yaw | held-out | LARGE | 168 | 4235 | 19.4, 16.3, 64.4 | 0.23, 0.24, 0.54 |
| full+mask-yaw | absent | SMALL | 32 | 1275 | 36.5, 17.2, 46.3 | 0.34, 0.26, 0.41 |
| full+mask-yaw | absent | MID | 1 | 50 | 20.0, 26.0, 54.0 | 0.24, 0.27, 0.49 |
| full+mask-yaw | absent | LARGE | 3 | 61 | 16.4, 14.8, 68.9 | 0.21, 0.23, 0.56 |

### commaCarSegments (`ccs`): unlabelled powertrains (HEV / PHEV / FCEV / MIXED platforms), predicted BEV share per platform (full model)

Scored but outside the BEV / ICE metrics: share of the platform's windows and devices called BEV and the mean P(BEV) (MIXED = the platform merges gasoline and electrified variants upstream, as in training).

| platform | model | powertrain | stratum | veh | win | windows called BEV % | devices called BEV | mean P(BEV) |
|---|---|---|---|---|---|---|---|---|
| AUDI_A3_MK3 | Audi A3 | MIXED | seen | 6 | 237 | 7.6 | 0/6 | 0.14 |
| CHEVROLET_VOLT | Chevrolet Volt | PHEV | seen | 24 | 386 | 40.2 | 6/24 | 0.42 |
| CHEVROLET_VOLT_2019 | Chevrolet Volt | PHEV | seen | 1 | 5 | 80.0 | 1/1 | 0.66 |
| CHRYSLER_PACIFICA_2018_HYBRID | Chrysler Pacifica | PHEV | seen | 29 | 378 | 33.3 | 3/29 | 0.36 |
| CHRYSLER_PACIFICA_2019_HYBRID | Chrysler Pacifica | PHEV | seen | 96 | 388 | 38.7 | 21/96 | 0.40 |
| FORD_ESCAPE_MK4 | Ford Escape | MIXED | seen | 13 | 269 | 36.8 | 5/13 | 0.39 |
| FORD_ESCAPE_MK4_5 | Ford Escape | MIXED | seen | 1 | 1 | 0.0 | 0/1 | 0.03 |
| FORD_EXPLORER_MK6 | Ford Explorer | MIXED | seen | 44 | 374 | 23.3 | 3/44 | 0.28 |
| FORD_F_150_MK14 | Ford F-150 | MIXED | seen | 42 | 387 | 15.0 | 1/42 | 0.21 |
| FORD_MAVERICK_MK1 | Ford Maverick | MIXED | seen | 32 | 382 | 29.3 | 4/32 | 0.32 |
| HONDA_ACCORD | Honda Accord | HEV | seen | 120 | 390 | 9.7 | 4/120 | 0.19 |
| HONDA_CRV_HYBRID | Honda CR-V | HEV | seen | 13 | 304 | 13.2 | 1/13 | 0.26 |
| HONDA_INSIGHT | Honda Insight | HEV | seen | 15 | 334 | 9.9 | 1/15 | 0.19 |
| HYUNDAI_AZERA_HEV_6TH_GEN | Hyundai Azera | HEV | held-out | 1 | 5 | 80.0 | 1/1 | 0.66 |
| HYUNDAI_ELANTRA_HEV_2021 | Hyundai Elantra | HEV | seen | 11 | 334 | 56.6 | 6/11 | 0.54 |
| HYUNDAI_IONIQ | Hyundai Ioniq | HEV | held-out | 4 | 104 | 26.9 | 0/4 | 0.30 |
| HYUNDAI_IONIQ_HEV_2022 | Hyundai Ioniq | HEV | held-out | 8 | 304 | 51.0 | 3/8 | 0.50 |
| HYUNDAI_IONIQ_PHEV | Hyundai Ioniq | PHEV | held-out | 9 | 292 | 51.0 | 4/9 | 0.50 |
| HYUNDAI_IONIQ_PHEV_2019 | Hyundai Ioniq Plug-in | PHEV | absent | 2 | 100 | 39.0 | 1/2 | 0.42 |
| HYUNDAI_SANTA_FE_HEV_2022 | Hyundai Santa Fe | HEV | held-out | 8 | 262 | 45.0 | 3/8 | 0.47 |
| HYUNDAI_SANTA_FE_PHEV_2022 | Hyundai Santa Fe | PHEV | held-out | 8 | 301 | 52.2 | 5/8 | 0.51 |
| HYUNDAI_SONATA_HYBRID | Hyundai Sonata | HEV | seen | 13 | 328 | 65.9 | 10/13 | 0.62 |
| HYUNDAI_TUCSON_4TH_GEN | Hyundai Tucson | MIXED | seen | 27 | 321 | 24.9 | 5/27 | 0.31 |
| KIA_NIRO_HEV_2021 | Kia Niro | HEV | seen | 4 | 144 | 42.4 | 1/4 | 0.46 |
| KIA_NIRO_HEV_2ND_GEN | Kia Niro | HEV | seen | 8 | 187 | 38.5 | 0/8 | 0.42 |
| KIA_NIRO_PHEV | Kia Niro | PHEV | seen | 3 | 63 | 23.8 | 0/3 | 0.33 |
| KIA_NIRO_PHEV_2022 | Kia Niro | PHEV | seen | 2 | 95 | 48.4 | 0/2 | 0.48 |
| KIA_SORENTO_HEV_4TH_GEN | Kia Sorento | HEV | held-out | 8 | 287 | 16.7 | 0/8 | 0.27 |
| KIA_SPORTAGE_5TH_GEN | Kia Sportage | MIXED | seen | 9 | 277 | 23.5 | 1/9 | 0.28 |
| LEXUS_CTH | Lexus CT 200h | HEV | absent | 3 | 65 | 30.8 | 1/3 | 0.35 |
| LEXUS_ES_TSS2 | Lexus ES | HEV | seen | 62 | 375 | 37.3 | 25/62 | 0.40 |
| LEXUS_RX | Lexus RX | HEV | held-out | 55 | 968 | 20.0 | 4/55 | 0.23 |
| TOYOTA_ALPHARD_TSS2 | Toyota Alphard | MIXED | seen | 6 | 165 | 20.6 | 0/6 | 0.25 |
| TOYOTA_AVALON_2019 | Toyota Avalon | HEV | held-out | 25 | 774 | 34.4 | 6/25 | 0.38 |
| TOYOTA_AVALON_TSS2 | Toyota Avalon | unknown | held-out | 6 | 237 | 19.8 | 0/6 | 0.24 |
| TOYOTA_CAMRY | Toyota Camry | HEV | seen | 86 | 390 | 14.6 | 7/86 | 0.21 |
| TOYOTA_CAMRY_TSS2 | Toyota Camry | HEV | seen | 81 | 391 | 37.1 | 24/81 | 0.39 |
| TOYOTA_CHR | Toyota C-HR | MIXED | seen | 7 | 156 | 17.3 | 0/7 | 0.24 |
| TOYOTA_CHR_TSS2 | Toyota C-HR | MIXED | seen | 2 | 17 | 17.6 | 0/2 | 0.21 |
| TOYOTA_COROLLA_TSS2 | Toyota Corolla | HEV | held-out | 233 | 971 | 31.9 | 60/233 | 0.34 |
| TOYOTA_MIRAI | Toyota Mirai | FCEV | absent | 6 | 296 | 36.5 | 0/6 | 0.41 |
| TOYOTA_PRIUS | Toyota Prius | HEV | seen | 109 | 368 | 37.0 | 32/109 | 0.41 |
| TOYOTA_PRIUS_TSS2 | Toyota Prius | HEV | seen | 99 | 399 | 46.4 | 39/99 | 0.47 |
| TOYOTA_PRIUS_V | Toyota Prius v | HEV | absent | 3 | 50 | 60.0 | 2/3 | 0.57 |
| TOYOTA_RAV4H | Toyota RAV4 | HEV | seen | 37 | 389 | 33.2 | 5/37 | 0.35 |
| TOYOTA_RAV4_PRIME | Toyota RAV4 | PHEV | seen | 40 | 384 | 59.1 | 28/40 | 0.56 |
| TOYOTA_RAV4_TSS2_2022 | Toyota RAV4 | HEV | seen | 67 | 384 | 25.0 | 12/67 | 0.32 |
| TOYOTA_SIENNA_4TH_GEN | Toyota Sienna | HEV | held-out | 32 | 879 | 33.6 | 2/32 | 0.36 |
| VOLKSWAGEN_ARTEON_MK1 | Volkswagen Arteon | MIXED | seen | 4 | 134 | 18.7 | 0/4 | 0.28 |
| VOLKSWAGEN_CRAFTER_MK2 | Volkswagen Crafter | MIXED | seen | 4 | 115 | 22.6 | 1/4 | 0.25 |
| VOLKSWAGEN_GOLF_MK7 | Volkswagen Golf | MIXED | seen | 40 | 391 | 12.3 | 0/40 | 0.20 |
| VOLKSWAGEN_PASSAT_MK8 | Volkswagen Passat | MIXED | held-out | 8 | 256 | 16.4 | 2/8 | 0.21 |
| VOLKSWAGEN_TIGUAN_MK2 | Volkswagen Tiguan | MIXED | seen | 31 | 385 | 11.4 | 1/31 | 0.17 |

### commaCarSegments (`ccs_honda`): decisions per platform, full 20-channel models (V1 / V3 / V2 variants for comparison)

One row per openpilot platform; the devices are disjoint from the lab library (vehicle = device pseudonym, drive = device x route, one window per segment). Sorted by stratum (seen = training / validation model of the lab's `split_platform` split, held-out = one of the 41 test models, absent = not in the lab inventory), then footprint. `label` = source of footprint and model (lab table / public specs); `lead` = share of the windows per lead source (radar tracks / stock-ACC object / none = vision-only platform); `DAS %` = mean share of automation-engaged samples (driving automation system: stock cruise or controls allowed). Model-level decision = mean probability over the platform's labelled windows -> arg-max (`OK` / `MISS`); `veh right` = devices whose mean probability picks the label; HEV / PHEV / FCEV / MIXED platforms are unlabelled for the powertrain task and show the BEV probability and the predicted class only.

| platform | model | footprint | size | powertrain | stratum | label | lead % | veh | drives | win | DAS % | size full: P(S, M, L) % -> decision | size veh right | size drive acc % | size V1 | size V3 | BEV vs ICE full: P(BEV) % -> decision | pt veh right | pt V3 | pt V2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HONDA_FIT | Honda Fit | 3.74 | SMALL | ICE | seen | lab | radar 99 / none 1 | 7 | 116 | 273 | 35.0 | 60, 36, 4 -> SMALL OK | 6/7 | 73.3 | MID MISS | SMALL OK | 17 -> ICE OK | 7/7 | ICE OK | ICE OK |
| HONDA_CIVIC | Honda Civic | 4.20 | SMALL | ICE | seen | lab | radar 98 / none 2 | 33 | 186 | 385 | 45.7 | 66, 30, 4 -> SMALL OK | 32/33 | 83.3 | MID MISS | SMALL OK | 18 -> ICE OK | 33/33 | ICE OK | ICE OK |
| HONDA_CIVIC_BOSCH | Honda Civic | 4.20 | SMALL | ICE | seen | lab | none 100 | 55 | 154 | 395 | 39.7 | 75, 21, 4 -> SMALL OK | 55/55 | 96.1 | SMALL OK | SMALL OK | 11 -> ICE OK | 55/55 | ICE OK | ICE OK |
| HONDA_CIVIC_2022 | Honda Civic | 4.21 | SMALL | ICE | seen | lab | none 100 | 36 | 148 | 393 | 45.6 | 67, 28, 4 -> SMALL OK | 35/36 | 87.8 | SMALL OK | SMALL OK | 21 -> ICE OK | 34/36 | ICE OK | ICE OK |
| HONDA_CRV | Honda CR-V | 4.23 | SMALL | ICE | seen | lab | radar 89 / none 11 | 4 | 46 | 106 | 43.6 | 79, 19, 3 -> SMALL OK | 4/4 | 93.5 | SMALL OK | SMALL OK | 25 -> ICE OK | 4/4 | ICE OK | ICE OK |
| HONDA_CRV_5G | Honda CR-V | 4.28 | SMALL | ICE | seen | lab | none 100 | 92 | 154 | 380 | 45.4 | 75, 20, 4 -> SMALL OK | 91/92 | 93.5 | MID MISS | SMALL OK | 15 -> ICE OK | 91/92 | ICE OK | ICE OK |
| HONDA_HRV | Honda HR-V | 4.01 | SMALL | ICE | held-out | lab | radar 99 / none 1 | 2 | 62 | 97 | 20.4 | 74, 23, 3 -> SMALL OK | 2/2 | 96.8 | MID MISS | SMALL OK | 16 -> ICE OK | 2/2 | ICE OK | ICE OK |
| HONDA_HRV_3G | Honda HR-V | 4.17 | SMALL | ICE | held-out | lab | none 100 | 3 | 17 | 55 | 56.2 | 73, 25, 3 -> SMALL OK | 3/3 | 94.1 | MID MISS | SMALL OK | 17 -> ICE OK | 3/3 | ICE OK | ICE OK |
| HONDA_FREED | Honda Freed | 4.06 | SMALL | unlabelled: unknown | absent | public | radar 100 | 1 | 30 | 48 | 25.6 | 55, 34, 10 -> SMALL OK | 1/1 | 80.0 | SMALL OK | SMALL OK | 20 -> ICE (unlabelled) | - | ICE (unlabelled) | ICE (unlabelled) |

### commaCarSegments (`ccs_honda`): pooled per stratum, every run (window / drive / vehicle = device / model = car model)

Accuracy and macro-F1 in percent; the vehicle and model levels carry 95 % intervals from a vehicle-cluster bootstrap of 300 resamples over the stratum's devices (score.py `level_eval` on the stratum subset; the pooled score files use 1,000). `units right per class` = diagonal of the confusion matrix / support; chance = 1 / K and the majority share of the level. Runs: full = the 20-channel models; V1 / V2 / V3 = the reduced-input variants; full+mask-Vk = the full models with that preset masked; full+mask-yaw / full+mask-lead = the sensitivity runs (steering-derived yaw group masked; lead group masked).

size (SMALL / MID / LARGE):

| run | stratum | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | units right per class | chance uniform / majority % |
|---|---|---|---|---|---|---|---|---|
| full | all | window | 2132 | 83.8 | - | SMALL 83.8 / MID - / LARGE - | SMALL 1786/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 88.2 | - | SMALL 88.2 / MID - / LARGE - | SMALL 805/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 98.3 [96.5, 99.6] | - | SMALL 98.3 / MID - / LARGE - | SMALL 226/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 83.5 | - | SMALL 83.5 / MID - / LARGE - | SMALL 1613/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 87.7 | - | SMALL 87.7 / MID - / LARGE - | SMALL 705/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 98.2 [96.4, 99.8] | - | SMALL 98.2 / MID - / LARGE - | SMALL 220/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 3/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 90.1 | - | SMALL 90.1 / MID - / LARGE - | SMALL 137/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 96.2 | - | SMALL 96.2 / MID - / LARGE - | SMALL 76/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 75.0 | - | SMALL 75.0 / MID - / LARGE - | SMALL 36/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 80.0 | - | SMALL 80.0 / MID - / LARGE - | SMALL 24/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
| V1 | all | window | 2132 | 49.6 | - | SMALL 49.6 / MID - / LARGE - | SMALL 1057/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 49.5 | - | SMALL 49.5 / MID - / LARGE - | SMALL 452/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 47.8 [41.5, 54.8] | - | SMALL 47.8 / MID - / LARGE - | SMALL 110/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 60.0 [20.0, 60.0] | - | SMALL 60.0 / MID - / LARGE - | SMALL 3/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 49.7 | - | SMALL 49.7 / MID - / LARGE - | SMALL 961/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 49.0 | - | SMALL 49.0 / MID - / LARGE - | SMALL 394/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 48.2 [41.3, 54.9] | - | SMALL 48.2 / MID - / LARGE - | SMALL 108/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 66.7 [0.0, 66.7] | - | SMALL 66.7 / MID - / LARGE - | SMALL 2/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 42.1 | - | SMALL 42.1 / MID - / LARGE - | SMALL 64/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 44.3 | - | SMALL 44.3 / MID - / LARGE - | SMALL 35/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 20.0 [0.0, 60.0] | - | SMALL 20.0 / MID - / LARGE - | SMALL 1/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 66.7 | - | SMALL 66.7 / MID - / LARGE - | SMALL 32/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 76.7 | - | SMALL 76.7 / MID - / LARGE - | SMALL 23/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
| V2 | all | window | 2132 | 51.4 | - | SMALL 51.4 / MID - / LARGE - | SMALL 1096/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 53.6 | - | SMALL 53.6 / MID - / LARGE - | SMALL 489/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 73.0 [68.3, 79.1] | - | SMALL 73.0 / MID - / LARGE - | SMALL 168/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 60.0 [40.0, 100.0] | - | SMALL 60.0 / MID - / LARGE - | SMALL 3/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 52.3 | - | SMALL 52.3 / MID - / LARGE - | SMALL 1011/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 55.6 | - | SMALL 55.6 / MID - / LARGE - | SMALL 447/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 74.1 [68.5, 79.0] | - | SMALL 74.1 / MID - / LARGE - | SMALL 166/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 66.7 [66.7, 100.0] | - | SMALL 66.7 / MID - / LARGE - | SMALL 2/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 45.4 | - | SMALL 45.4 / MID - / LARGE - | SMALL 69/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 36.7 | - | SMALL 36.7 / MID - / LARGE - | SMALL 29/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 40.0 [0.0, 80.0] | - | SMALL 40.0 / MID - / LARGE - | SMALL 2/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 100.0 [0.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 33.3 | - | SMALL 33.3 / MID - / LARGE - | SMALL 16/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 43.3 | - | SMALL 43.3 / MID - / LARGE - | SMALL 13/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
| V3 | all | window | 2132 | 64.2 | - | SMALL 64.2 / MID - / LARGE - | SMALL 1368/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 72.2 | - | SMALL 72.2 / MID - / LARGE - | SMALL 659/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 86.5 [82.2, 90.9] | - | SMALL 86.5 / MID - / LARGE - | SMALL 199/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 63.6 | - | SMALL 63.6 / MID - / LARGE - | SMALL 1228/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 71.8 | - | SMALL 71.8 / MID - / LARGE - | SMALL 577/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 86.2 [81.9, 90.2] | - | SMALL 86.2 / MID - / LARGE - | SMALL 193/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 3/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 68.4 | - | SMALL 68.4 / MID - / LARGE - | SMALL 104/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 69.6 | - | SMALL 69.6 / MID - / LARGE - | SMALL 55/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 75.0 | - | SMALL 75.0 / MID - / LARGE - | SMALL 36/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 90.0 | - | SMALL 90.0 / MID - / LARGE - | SMALL 27/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
| full+mask-V1 | all | window | 2132 | 92.5 | - | SMALL 92.5 / MID - / LARGE - | SMALL 1972/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 95.5 | - | SMALL 95.5 / MID - / LARGE - | SMALL 872/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 98.7 [97.2, 99.8] | - | SMALL 98.7 / MID - / LARGE - | SMALL 227/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 92.4 | - | SMALL 92.4 / MID - / LARGE - | SMALL 1785/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 95.3 | - | SMALL 95.3 / MID - / LARGE - | SMALL 766/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 98.7 [97.3, 100.0] | - | SMALL 98.7 / MID - / LARGE - | SMALL 221/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 3/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 96.7 | - | SMALL 96.7 / MID - / LARGE - | SMALL 147/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 98.7 | - | SMALL 98.7 / MID - / LARGE - | SMALL 78/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 83.3 | - | SMALL 83.3 / MID - / LARGE - | SMALL 40/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 93.3 | - | SMALL 93.3 / MID - / LARGE - | SMALL 28/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
| full+mask-V2 | all | window | 2132 | 81.7 | - | SMALL 81.7 / MID - / LARGE - | SMALL 1742/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 86.2 | - | SMALL 86.2 / MID - / LARGE - | SMALL 787/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 96.5 [93.9, 98.9] | - | SMALL 96.5 / MID - / LARGE - | SMALL 222/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 82.7 | - | SMALL 82.7 / MID - / LARGE - | SMALL 1597/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 87.8 | - | SMALL 87.8 / MID - / LARGE - | SMALL 706/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 96.4 [93.8, 98.2] | - | SMALL 96.4 / MID - / LARGE - | SMALL 216/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 3/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 78.3 | - | SMALL 78.3 / MID - / LARGE - | SMALL 119/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 77.2 | - | SMALL 77.2 / MID - / LARGE - | SMALL 61/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 54.2 | - | SMALL 54.2 / MID - / LARGE - | SMALL 26/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 66.7 | - | SMALL 66.7 / MID - / LARGE - | SMALL 20/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
| full+mask-V3 | all | window | 2132 | 89.5 | - | SMALL 89.5 / MID - / LARGE - | SMALL 1909/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 93.2 | - | SMALL 93.2 / MID - / LARGE - | SMALL 851/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 97.0 [94.3, 98.7] | - | SMALL 97.0 / MID - / LARGE - | SMALL 223/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 89.1 | - | SMALL 89.1 / MID - / LARGE - | SMALL 1722/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 93.0 | - | SMALL 93.0 / MID - / LARGE - | SMALL 748/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 96.9 [94.2, 99.1] | - | SMALL 96.9 / MID - / LARGE - | SMALL 217/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 3/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 91.4 | - | SMALL 91.4 / MID - / LARGE - | SMALL 139/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 92.4 | - | SMALL 92.4 / MID - / LARGE - | SMALL 73/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 100.0 | - | SMALL 100.0 / MID - / LARGE - | SMALL 48/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 100.0 | - | SMALL 100.0 / MID - / LARGE - | SMALL 30/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
| full+mask-yaw | all | window | 2132 | 20.2 | - | SMALL 20.2 / MID - / LARGE - | SMALL 431/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 21.7 | - | SMALL 21.7 / MID - / LARGE - | SMALL 198/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 14.3 [9.8, 19.4] | - | SMALL 14.3 / MID - / LARGE - | SMALL 33/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 20.8 | - | SMALL 20.8 / MID - / LARGE - | SMALL 402/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 22.4 | - | SMALL 22.4 / MID - / LARGE - | SMALL 180/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 14.7 [10.5, 19.2] | - | SMALL 14.7 / MID - / LARGE - | SMALL 33/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 15.1 | - | SMALL 15.1 / MID - / LARGE - | SMALL 23/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 16.5 | - | SMALL 16.5 / MID - / LARGE - | SMALL 13/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 12.5 | - | SMALL 12.5 / MID - / LARGE - | SMALL 6/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 16.7 | - | SMALL 16.7 / MID - / LARGE - | SMALL 5/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 0.0 [0.0, 0.0] | - | SMALL 0.0 / MID - / LARGE - | SMALL 0/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
| full+mask-lead | all | window | 2132 | 86.2 | - | SMALL 86.2 / MID - / LARGE - | SMALL 1838/2132 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | drive | 913 | 91.1 | - | SMALL 91.1 / MID - / LARGE - | SMALL 832/913 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | vehicle | 230 | 98.7 [97.0, 100.0] | - | SMALL 98.7 / MID - / LARGE - | SMALL 227/230 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | all | model | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | window | 1932 | 85.6 | - | SMALL 85.6 / MID - / LARGE - | SMALL 1654/1932 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | drive | 804 | 90.3 | - | SMALL 90.3 / MID - / LARGE - | SMALL 726/804 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | vehicle | 224 | 98.7 [96.9, 100.0] | - | SMALL 98.7 / MID - / LARGE - | SMALL 221/224 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 3/3 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | window | 152 | 93.4 | - | SMALL 93.4 / MID - / LARGE - | SMALL 142/152 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | drive | 79 | 98.7 | - | SMALL 98.7 / MID - / LARGE - | SMALL 78/79 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 5/5 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | window | 48 | 87.5 | - | SMALL 87.5 / MID - / LARGE - | SMALL 42/48 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | drive | 30 | 93.3 | - | SMALL 93.3 / MID - / LARGE - | SMALL 28/30 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | vehicle | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |
|  | absent | model | 1 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | SMALL 1/1 / MID 0/0 / LARGE 0/0 | 33.3 / 100.0 |

bev_ice (BEV / ICE):

| run | stratum | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | units right per class | chance uniform / majority % |
|---|---|---|---|---|---|---|---|---|
| full | all | window | 2084 | 89.3 | - | BEV - / ICE 89.3 | BEV 0/0 / ICE 1860/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 94.1 | - | BEV - / ICE 94.1 | BEV 0/0 / ICE 831/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 99.1 [97.8, 100.0] | - | BEV - / ICE 99.1 | BEV 0/0 / ICE 227/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 89.1 | - | BEV - / ICE 89.1 | BEV 0/0 / ICE 1721/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 94.0 | - | BEV - / ICE 94.0 | BEV 0/0 / ICE 756/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 99.1 [97.8, 100.0] | - | BEV - / ICE 99.1 | BEV 0/0 / ICE 222/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 91.4 | - | BEV - / ICE 91.4 | BEV 0/0 / ICE 139/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 94.9 | - | BEV - / ICE 94.9 | BEV 0/0 / ICE 75/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 5/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |
| V1 | all | window | 2084 | 72.7 | - | BEV - / ICE 72.7 | BEV 0/0 / ICE 1515/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 79.7 | - | BEV - / ICE 79.7 | BEV 0/0 / ICE 704/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 86.0 [81.2, 89.7] | - | BEV - / ICE 86.0 | BEV 0/0 / ICE 197/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [75.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 72.2 | - | BEV - / ICE 72.2 | BEV 0/0 / ICE 1394/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 79.2 | - | BEV - / ICE 79.2 | BEV 0/0 / ICE 637/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 86.6 [82.4, 90.6] | - | BEV - / ICE 86.6 | BEV 0/0 / ICE 194/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 79.6 | - | BEV - / ICE 79.6 | BEV 0/0 / ICE 121/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 84.8 | - | BEV - / ICE 84.8 | BEV 0/0 / ICE 67/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 60.0 [20.0, 100.0] | - | BEV - / ICE 60.0 | BEV 0/0 / ICE 3/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |
| V2 | all | window | 2084 | 83.6 | - | BEV - / ICE 83.6 | BEV 0/0 / ICE 1743/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 88.7 | - | BEV - / ICE 88.7 | BEV 0/0 / ICE 783/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 97.8 [95.6, 99.6] | - | BEV - / ICE 97.8 | BEV 0/0 / ICE 224/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 83.1 | - | BEV - / ICE 83.1 | BEV 0/0 / ICE 1606/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 88.2 | - | BEV - / ICE 88.2 | BEV 0/0 / ICE 709/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 97.8 [96.0, 99.6] | - | BEV - / ICE 97.8 | BEV 0/0 / ICE 219/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 90.1 | - | BEV - / ICE 90.1 | BEV 0/0 / ICE 137/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 93.7 | - | BEV - / ICE 93.7 | BEV 0/0 / ICE 74/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 5/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |
| V3 | all | window | 2084 | 88.8 | - | BEV - / ICE 88.8 | BEV 0/0 / ICE 1850/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 94.5 | - | BEV - / ICE 94.5 | BEV 0/0 / ICE 834/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 98.3 [96.1, 99.6] | - | BEV - / ICE 98.3 | BEV 0/0 / ICE 225/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 88.3 | - | BEV - / ICE 88.3 | BEV 0/0 / ICE 1706/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 93.9 | - | BEV - / ICE 93.9 | BEV 0/0 / ICE 755/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 98.2 [96.4, 99.6] | - | BEV - / ICE 98.2 | BEV 0/0 / ICE 220/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 94.7 | - | BEV - / ICE 94.7 | BEV 0/0 / ICE 144/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 100.0 | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 79/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 5/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |
| full+mask-V1 | all | window | 2084 | 89.9 | - | BEV - / ICE 89.9 | BEV 0/0 / ICE 1874/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 93.0 | - | BEV - / ICE 93.0 | BEV 0/0 / ICE 821/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 98.3 [96.1, 99.6] | - | BEV - / ICE 98.3 | BEV 0/0 / ICE 225/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 89.4 | - | BEV - / ICE 89.4 | BEV 0/0 / ICE 1728/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 92.4 | - | BEV - / ICE 92.4 | BEV 0/0 / ICE 743/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 98.2 [96.4, 99.6] | - | BEV - / ICE 98.2 | BEV 0/0 / ICE 220/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 96.1 | - | BEV - / ICE 96.1 | BEV 0/0 / ICE 146/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 98.7 | - | BEV - / ICE 98.7 | BEV 0/0 / ICE 78/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 5/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |
| full+mask-V2 | all | window | 2084 | 86.3 | - | BEV - / ICE 86.3 | BEV 0/0 / ICE 1798/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 90.8 | - | BEV - / ICE 90.8 | BEV 0/0 / ICE 802/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 99.6 [98.7, 100.0] | - | BEV - / ICE 99.6 | BEV 0/0 / ICE 228/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 86.3 | - | BEV - / ICE 86.3 | BEV 0/0 / ICE 1668/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 91.2 | - | BEV - / ICE 91.2 | BEV 0/0 / ICE 733/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 99.6 [98.2, 100.0] | - | BEV - / ICE 99.6 | BEV 0/0 / ICE 223/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 85.5 | - | BEV - / ICE 85.5 | BEV 0/0 / ICE 130/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 87.3 | - | BEV - / ICE 87.3 | BEV 0/0 / ICE 69/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 5/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |
| full+mask-V3 | all | window | 2084 | 94.0 | - | BEV - / ICE 94.0 | BEV 0/0 / ICE 1959/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 95.8 | - | BEV - / ICE 95.8 | BEV 0/0 / ICE 846/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 229/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 93.6 | - | BEV - / ICE 93.6 | BEV 0/0 / ICE 1809/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 95.4 | - | BEV - / ICE 95.4 | BEV 0/0 / ICE 767/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 224/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 98.7 | - | BEV - / ICE 98.7 | BEV 0/0 / ICE 150/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 100.0 | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 79/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 5/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |
| full+mask-yaw | all | window | 2084 | 87.5 | - | BEV - / ICE 87.5 | BEV 0/0 / ICE 1823/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 91.5 | - | BEV - / ICE 91.5 | BEV 0/0 / ICE 808/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 98.7 [97.4, 99.6] | - | BEV - / ICE 98.7 | BEV 0/0 / ICE 226/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 87.4 | - | BEV - / ICE 87.4 | BEV 0/0 / ICE 1688/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 91.8 | - | BEV - / ICE 91.8 | BEV 0/0 / ICE 738/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 98.7 [96.9, 100.0] | - | BEV - / ICE 98.7 | BEV 0/0 / ICE 221/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 88.8 | - | BEV - / ICE 88.8 | BEV 0/0 / ICE 135/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 88.6 | - | BEV - / ICE 88.6 | BEV 0/0 / ICE 70/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 5/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |
| full+mask-lead | all | window | 2084 | 91.7 | - | BEV - / ICE 91.7 | BEV 0/0 / ICE 1911/2084 | 50.0 / 100.0 |
|  | all | drive | 883 | 95.8 | - | BEV - / ICE 95.8 | BEV 0/0 / ICE 846/883 | 50.0 / 100.0 |
|  | all | vehicle | 229 | 99.1 [97.8, 100.0] | - | BEV - / ICE 99.1 | BEV 0/0 / ICE 227/229 | 50.0 / 100.0 |
|  | all | model | 4 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 4/4 | 50.0 / 100.0 |
|  | seen | window | 1932 | 91.5 | - | BEV - / ICE 91.5 | BEV 0/0 / ICE 1768/1932 | 50.0 / 100.0 |
|  | seen | drive | 804 | 95.6 | - | BEV - / ICE 95.6 | BEV 0/0 / ICE 769/804 | 50.0 / 100.0 |
|  | seen | vehicle | 224 | 99.1 [97.8, 100.0] | - | BEV - / ICE 99.1 | BEV 0/0 / ICE 222/224 | 50.0 / 100.0 |
|  | seen | model | 3 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 3/3 | 50.0 / 100.0 |
|  | held-out | window | 152 | 94.1 | - | BEV - / ICE 94.1 | BEV 0/0 / ICE 143/152 | 50.0 / 100.0 |
|  | held-out | drive | 79 | 97.5 | - | BEV - / ICE 97.5 | BEV 0/0 / ICE 77/79 | 50.0 / 100.0 |
|  | held-out | vehicle | 5 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 5/5 | 50.0 / 100.0 |
|  | held-out | model | 1 | 100.0 [100.0, 100.0] | - | BEV - / ICE 100.0 | BEV 0/0 / ICE 1/1 | 50.0 / 100.0 |

### commaCarSegments (`ccs_honda`): held-out models against the internal per-model result (full models)

Every held-out `split_platform` test model present in the set: the external vehicle-level accuracy (devices right / devices) and the model-level decision with P(true), next to the internal result on the lab's own test windows of that model (internal_per_model.md, column `full`: drive accuracy % / decision (P true)). A model may span several platforms (e.g. gasoline and hybrid ports); `-` = no label for that task or no internal row.

| held-out model | platforms | veh | win | size label | size: veh right (acc %) | size: model decision (P true) | size: internal drive acc / decision (P true) | pt label | pt: veh right (acc %) | pt: model decision (P true) | pt: internal drive acc / decision (P true) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Honda HR-V | 2 | 5 | 152 | SMALL | 5/5 (100.0) | SMALL (0.73) OK | 46.0 / SMALL (0.50) | ICE | 5/5 (100.0) | ICE (0.84) OK | 95.0 / ICE (0.84) |

### commaCarSegments (`ccs_honda`): seen and absent models (full models; seen platform, new device)

Per car model of the seen stratum (training / validation models of the lab split, here on devices the lab never saw) and of the absent stratum (models outside the lab inventory): model-level hits, the median per-model vehicle accuracy, then the ten seen models with the lowest P(true) and every absent model.

size (SMALL / MID / LARGE), seen models: 3 / 3 right at model level; median per-model vehicle accuracy 98.4 %; devices right 222 / 226 (98.2 %).

| model | label | platforms | veh | win | veh right (acc %) | P(true) | model decision |
|---|---|---|---|---|---|---|---|
| Honda Fit | SMALL | 1 | 7 | 273 | 6/7 (85.7) | 0.60 | SMALL OK |
| Honda Civic | SMALL | 3 | 123 | 1173 | 121/123 (98.4) | 0.70 | SMALL OK |
| Honda CR-V | SMALL | 2 | 96 | 486 | 95/96 (99.0) | 0.76 | SMALL OK |

size (SMALL / MID / LARGE), absent models: 1 / 1 right at model level; median per-model vehicle accuracy 100.0 %; devices right 1 / 1 (100.0 %).

| model | label | platforms | veh | win | veh right (acc %) | P(true) | model decision |
|---|---|---|---|---|---|---|---|
| Honda Freed | SMALL | 1 | 1 | 48 | 1/1 (100.0) | 0.55 | SMALL OK |

bev_ice (BEV / ICE), seen models: 3 / 3 right at model level; median per-model vehicle accuracy 99.0 %; devices right 223 / 226 (98.7 %).

| model | label | platforms | veh | win | veh right (acc %) | P(true) | model decision |
|---|---|---|---|---|---|---|---|
| Honda CR-V | ICE | 2 | 96 | 486 | 95/96 (99.0) | 0.82 | ICE OK |
| Honda Fit | ICE | 1 | 7 | 273 | 7/7 (100.0) | 0.83 | ICE OK |
| Honda Civic | ICE | 3 | 123 | 1173 | 121/123 (98.4) | 0.83 | ICE OK |

### commaCarSegments (`ccs_honda`): lead source and automation state per stratum (full models)

`lead` = the platform's lead source (radar tracks / stock-ACC object / none = vision-only); `state`: human = das_frac < 0.5, engaged = das_frac >= 0.5 (the driving automation system or stock cruise active for most of the minute). Vehicle accuracy inside a cell uses the vehicle's windows of that cell only.

| factor | value | stratum | size: win / veh | size win acc % | size veh acc % (right/n) | pt: win / veh | pt win acc % | pt veh acc % (right/n) |
|---|---|---|---|---|---|---|---|---|
| lead | radar | all | 887 / 47 | 80.4 | 95.7 (45/47) | 839 / 46 | 87.4 | 100.0 (46/46) |
| lead | radar | seen | 743 / 44 | 79.4 | 95.5 (42/44) | 743 / 44 | 87.1 | 100.0 (44/44) |
| lead | radar | held-out | 96 / 2 | 90.6 | 100.0 (2/2) | 96 / 2 | 89.6 | 100.0 (2/2) |
| lead | radar | absent | 48 / 1 | 75.0 | 100.0 (1/1) | 0 / 0 | - | - |
| lead | none | all | 1245 / 193 | 86.2 | 97.9 (189/193) | 1245 / 193 | 90.5 | 97.9 (189/193) |
| lead | none | seen | 1189 / 189 | 86.0 | 98.4 (186/189) | 1189 / 189 | 90.3 | 97.9 (185/189) |
| lead | none | held-out | 56 / 4 | 89.3 | 75.0 (3/4) | 56 / 4 | 94.6 | 100.0 (4/4) |
| state | human | all | 1230 / 204 | 91.1 | 98.5 (201/204) | 1194 / 203 | 95.6 | 99.5 (202/203) |
| state | human | seen | 1094 / 198 | 91.0 | 98.5 (195/198) | 1094 / 198 | 95.6 | 99.5 (197/198) |
| state | human | held-out | 100 / 5 | 96.0 | 100.0 (5/5) | 100 / 5 | 96.0 | 100.0 (5/5) |
| state | human | absent | 36 / 1 | 80.6 | 100.0 (1/1) | 0 / 0 | - | - |
| state | engaged | all | 902 / 171 | 73.7 | 84.8 (145/171) | 890 / 170 | 80.7 | 94.1 (160/170) |
| state | engaged | seen | 838 / 165 | 73.6 | 84.8 (140/165) | 838 / 165 | 80.5 | 94.5 (156/165) |
| state | engaged | held-out | 52 / 5 | 78.8 | 80.0 (4/5) | 52 / 5 | 82.7 | 80.0 (4/5) |
| state | engaged | absent | 12 / 1 | 58.3 | 100.0 (1/1) | 0 / 0 | - | - |

### commaCarSegments (`ccs_honda`): evidence per vehicle (full models; vehicle-level accuracy by labelled windows per device)

How many minutes of a new device are needed: devices bucketed by their number of labelled windows (1, 2-4, 5-9, 10+); cells = devices right / devices (accuracy %).

| task | stratum | devices | 1 window | 2-4 windows | 5-9 windows | 10+ windows |
|---|---|---|---|---|---|---|
| size | all | 230 | 15/17 (88.2) | 78/79 (98.7) | 60/61 (98.4) | 73/73 (100.0) |
| size | seen | 224 | 15/17 (88.2) | 77/78 (98.7) | 59/60 (98.3) | 69/69 (100.0) |
| size | held-out | 5 | - | 1/1 (100.0) | 1/1 (100.0) | 3/3 (100.0) |
| size | absent | 1 | - | - | - | 1/1 (100.0) |
| bev_ice | all | 229 | 16/17 (94.1) | 79/79 (100.0) | 60/61 (98.4) | 72/72 (100.0) |
| bev_ice | seen | 224 | 16/17 (94.1) | 78/78 (100.0) | 59/60 (98.3) | 69/69 (100.0) |
| bev_ice | held-out | 5 | - | 1/1 (100.0) | 1/1 (100.0) | 3/3 (100.0) |

### commaCarSegments (`ccs_honda`): predicted-share confusion per stratum (size; full model and full model with the yaw group masked)

Predicted class shares of the windows per true class and stratum (the confusion matrix as shares) and the mean probabilities; the yaw-masked run shows how much of the decision rests on the steering-derived yaw / curvature / lateral-acceleration group.

| run | stratum | true class | veh | win | predicted share SMALL, MID, LARGE % | mean P(SMALL, MID, LARGE) |
|---|---|---|---|---|---|---|
| full | all | SMALL | 230 | 2132 | 83.8, 14.8, 1.5 | 0.70, 0.26, 0.04 |
| full | seen | SMALL | 224 | 1932 | 83.5, 15.2, 1.3 | 0.70, 0.26, 0.04 |
| full | held-out | SMALL | 5 | 152 | 90.1, 7.9, 2.0 | 0.73, 0.24, 0.03 |
| full | absent | SMALL | 1 | 48 | 75.0, 18.8, 6.2 | 0.55, 0.34, 0.10 |
| full+mask-yaw | all | SMALL | 230 | 2132 | 20.2, 9.1, 70.7 | 0.23, 0.19, 0.58 |
| full+mask-yaw | seen | SMALL | 224 | 1932 | 20.8, 9.3, 69.9 | 0.23, 0.19, 0.58 |
| full+mask-yaw | held-out | SMALL | 5 | 152 | 15.1, 7.9, 77.0 | 0.20, 0.20, 0.60 |
| full+mask-yaw | absent | SMALL | 1 | 48 | 12.5, 4.2, 83.3 | 0.16, 0.13, 0.71 |

### commaCarSegments (`ccs_honda`): unlabelled powertrains (HEV / PHEV / FCEV / MIXED platforms), predicted BEV share per platform (full model)

Scored but outside the BEV / ICE metrics: share of the platform's windows and devices called BEV and the mean P(BEV) (MIXED = the platform merges gasoline and electrified variants upstream, as in training).

| platform | model | powertrain | stratum | veh | win | windows called BEV % | devices called BEV | mean P(BEV) |
|---|---|---|---|---|---|---|---|---|
| HONDA_FREED | Honda Freed | unknown | absent | 1 | 48 | 16.7 | 0/1 | 0.20 |

## Per dataset

## ultraav: ULTra-AV, named-vehicle step3 subset

### ultraav: bev_ice (BEV / ICE)

Windows 2976 (1987 with a bev_ice label), vehicles 22, drives 121, car models 20; label counts {'BEV': 1330, 'ICE': 657}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (longitudinal-only, 11 ch) | window | 1987 | 72.1 [54.8, 86.6] | 63.3 [51.0, 72.9] | 50.0 / 66.9 | BEV 90.5 / ICE 34.9 | 76.7 / 74.1; das 75.3 / 73.0 |
|  | drive | 84 | 85.7 [46.1, 96.7] | 46.2 [31.5, 49.1] | 50.0 / 89.3 | BEV 96.0 / ICE 0.0 | 90.0 / 89.0; das 86.9 / 86.3 |
|  | vehicle | 12 | 75.0 [50.0, 100.0] | 69.7 [36.8, 100.0] | 50.0 / 58.3 | BEV 100.0 / ICE 40.0 | 91.4 / 90.4; das 85.4 / 84.2 |
|  | model | 11 | 72.7 [57.1, 100.0] | 68.6 [38.5, 100.0] | 50.0 / 54.5 | BEV 100.0 / ICE 40.0 |  |
| V3 variant (speed-only, 3 ch) | window | 1987 | 52.0 [33.8, 71.2] | 51.4 [32.6, 65.2] | 50.0 / 66.9 | BEV 30.6 / ICE 95.3 | 77.0 / 74.6; das 75.1 / 73.4 |
|  | drive | 84 | 25.0 [14.3, 53.3] | 24.9 [13.9, 46.5] | 50.0 / 89.3 | BEV 16.0 / ICE 100.0 | 89.5 / 88.6; das 88.3 / 87.9 |
|  | vehicle | 12 | 41.7 [16.7, 66.7] | 29.4 [14.3, 40.0] | 50.0 / 58.3 | BEV 0.0 / ICE 100.0 | 91.0 / 90.1; das 86.5 / 85.6 |
|  | model | 11 | 45.5 [20.0, 66.7] | 31.2 [16.7, 40.0] | 50.0 / 54.5 | BEV 0.0 / ICE 100.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 1987 | 45.4 [33.6, 59.3] | 45.4 [31.9, 56.5] | 50.0 / 66.9 | BEV 32.3 / ICE 72.0 | 71.0 / 60.9; das 71.4 / 60.9 |
|  | drive | 84 | 23.8 [15.7, 50.0] | 23.6 [14.4, 42.3] | 50.0 / 89.3 | BEV 16.0 / ICE 88.9 | 67.5 / 54.4; das 66.0 / 53.7 |
|  | vehicle | 12 | 50.0 [25.0, 75.0] | 43.8 [20.0, 73.3] | 50.0 / 58.3 | BEV 14.3 / ICE 100.0 | 69.3 / 56.8; das 66.7 / 51.2 |
|  | model | 11 | 54.5 [28.6, 75.0] | 47.6 [22.2, 66.7] | 50.0 / 54.5 | BEV 16.7 / ICE 100.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 1987 | 35.1 [18.0, 56.2] | 32.4 [18.0, 44.8] | 50.0 / 66.9 | BEV 11.1 / ICE 83.7 | 69.7 / 59.4; das 70.3 / 62.0 |
|  | drive | 84 | 13.1 [3.4, 45.7] | 12.5 [3.7, 32.2] | 50.0 / 89.3 | BEV 2.7 / ICE 100.0 | 63.3 / 51.0; das 69.4 / 60.7 |
|  | vehicle | 12 | 41.7 [16.7, 66.7] | 29.4 [14.3, 40.0] | 50.0 / 58.3 | BEV 0.0 / ICE 100.0 | 65.2 / 49.8; das 70.3 / 59.0 |
|  | model | 11 | 45.5 [20.0, 66.7] | 31.2 [16.7, 40.0] | 50.0 / 54.5 | BEV 0.0 / ICE 100.0 |  |

Per car model, primary run (V2 variant (longitudinal-only, 11 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Audi A6 | ICE | 4.747 |  | ICE | 1 | 8 | 173 | 100.0 | 0.30 | 15.6 | 0.0 | 0/1 | BEV (0.70, 0.30) MISS |
| Audi e-tron | BEV | 4.841 | yes | BEV | 1 | 50 | 219 | 95.4 | 0.87 | 100.0 | 100.0 | 1/1 | BEV (0.88, 0.12) OK |
| BMW X5 | ICE | 4.985 |  | ICE | 1 | 8 | 173 | 100.0 | 0.40 | 32.9 | 12.5 | 0/1 | BEV (0.60, 0.40) MISS |
| Ford S-Max | ICE | 4.575 |  | ICE | 1 | 7 | 59 | 100.0 | 0.52 | 55.9 | 42.9 | 1/1 | ICE (0.48, 0.52) OK |
| Jaguar I-Pace | BEV | 4.917 | yes | BEV | 1 | 46 | 195 | 95.9 | 0.85 | 95.9 | 97.8 | 1/1 | BEV (0.85, 0.15) OK |
| Mazda 3 | ICE | 4.205 |  | ICE | 1 | 20 | 79 | 100.0 | 0.31 | 16.5 | 10.0 | 0/1 | BEV (0.69, 0.30) MISS |
| Mercedes-Benz A-Class | ICE | 4.249 |  | ICE | 1 | 8 | 173 | 100.0 | 0.52 | 57.2 | 50.0 | 1/1 | ICE (0.48, 0.52) OK |
| Tesla Model 3 | BEV | 4.571 |  | BEV | 2 | 53 | 388 | 96.9 | 0.77 | 90.5 | 94.3 | 2/2 | BEV (0.77, 0.23) OK |
| Tesla Model S | BEV | 4.973 |  | BEV | 1 | 62 | 255 | 95.7 | 0.79 | 91.8 | 91.9 | 1/1 | BEV (0.79, 0.21) OK |
| Tesla Model X | BEV | 5.061 |  | BEV | 1 | 50 | 218 | 98.2 | 0.74 | 85.3 | 94.0 | 1/1 | BEV (0.74, 0.26) OK |
| Volkswagen e-Golf | BEV | 4.020 |  | BEV | 1 | 6 | 55 | 100.0 | 0.53 | 49.1 | 83.3 | 1/1 | BEV (0.53, 0.47) OK |

Windows outside the bev_ice classes (scored, not in the metrics), primary run:

| model | label | veh | win | predicted share % | mean P |
|---|---|---|---|---|---|
| Audi A4 Avant | HEV | 1 | 154 | BEV 81.8, ICE 18.2 | 0.72, 0.28 |
| BMW i3s REx | PHEV | 1 | 236 | BEV 78.8, ICE 21.2 | 0.69, 0.31 |
| Ford Fusion | unknown | 1 | 1 | BEV 100.0, ICE 0.0 | 0.80, 0.20 |
| Hyundai Nexo | FCEV | 1 | 51 | BEV 86.3, ICE 13.7 | 0.71, 0.29 |
| Kia Niro PHEV | PHEV | 1 | 64 | BEV 53.1, ICE 46.9 | 0.50, 0.50 |
| Lincoln MKZ | unknown | 2 | 53 | BEV 30.2, ICE 69.8 | 0.39, 0.61 |
| Mercedes-Benz GLE 450 4MATIC | HEV | 1 | 272 | BEV 41.9, ICE 58.1 | 0.44, 0.56 |
| Mitsubishi Outlander PHEV | PHEV | 1 | 53 | BEV 37.7, ICE 62.3 | 0.40, 0.60 |
| Toyota RAV4 Hybrid | HEV | 1 | 105 | BEV 82.9, ICE 17.1 | 0.73, 0.27 |

### ultraav: size (SMALL / MID / LARGE)

Windows 2976 (2976 with a size label), vehicles 22, drives 121, car models 20; label counts {'SMALL': 919, 'MID': 944, 'LARGE': 1113}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (longitudinal-only, 11 ch) | window | 2976 | 29.7 [19.8, 41.3] | 27.9 [19.5, 35.5] | 33.3 / 37.4 | SMALL 34.2 / MID 50.8 / LARGE 8.2 | 41.7 / 34.9; das 43.0 / 35.0 |
|  | drive | 121 | 45.5 [30.5, 68.7] | 37.1 [26.7, 41.2] | 33.3 / 40.5 | SMALL 58.3 / MID 83.7 / LARGE 0.0 | 45.2 / 33.5; das 46.8 / 34.3 |
|  | vehicle | 22 | 45.5 [27.3, 68.2] | 35.7 [21.1, 48.9] | 33.3 / 40.9 | SMALL 50.0 / MID 66.7 / LARGE 0.0 | 43.2 / 32.8; das 46.2 / 36.0 |
|  | model | 20 | 45.0 [28.6, 63.6] | 36.3 [23.2, 48.5] | 33.3 / 40.0 | SMALL 50.0 / MID 71.4 / LARGE 0.0 |  |
| V3 variant (speed-only, 3 ch) | window | 2976 | 29.7 [14.0, 49.4] | 19.1 [10.9, 26.1] | 33.3 / 37.4 | SMALL 88.8 / MID 5.5 / LARGE 1.5 | 47.1 / 41.3; das 47.3 / 40.1 |
|  | drive | 121 | 21.5 [8.5, 33.3] | 14.0 [6.1, 18.8] | 33.3 / 40.5 | SMALL 100.0 / MID 4.1 / LARGE 0.0 | 53.4 / 45.1; das 49.4 / 39.3 |
|  | vehicle | 22 | 36.4 [18.2, 54.5] | 17.8 [10.3, 24.8] | 33.3 / 40.9 | SMALL 100.0 / MID 0.0 / LARGE 0.0 | 54.4 / 45.6; das 56.1 / 47.0 |
|  | model | 20 | 40.0 [23.1, 53.8] | 19.0 [12.5, 23.3] | 33.3 / 40.0 | SMALL 100.0 / MID 0.0 / LARGE 0.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 2976 | 28.8 [15.1, 47.8] | 26.0 [15.4, 38.6] | 33.3 / 37.4 | SMALL 62.4 / MID 14.7 / LARGE 13.0 | 53.6 / 33.2; das 57.2 / 37.9 |
|  | drive | 121 | 19.8 [5.1, 30.3] | 18.1 [4.6, 27.1] | 33.3 / 40.5 | SMALL 66.7 / MID 10.2 / LARGE 6.2 | 52.7 / 24.8; das 56.6 / 28.3 |
|  | vehicle | 22 | 22.7 [4.5, 40.9] | 17.5 [4.4, 30.8] | 33.3 / 40.9 | SMALL 50.0 / MID 0.0 / LARGE 20.0 | 48.0 / 23.2; das 51.1 / 26.3 |
|  | model | 20 | 25.0 [8.3, 38.5] | 18.5 [6.1, 27.8] | 33.3 / 40.0 | SMALL 50.0 / MID 0.0 / LARGE 20.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 2976 | 31.3 [13.2, 52.8] | 18.8 [10.0, 28.7] | 33.3 / 37.4 | SMALL 95.6 / MID 3.2 / LARGE 2.1 | 55.1 / 30.1; das 59.4 / 30.7 |
|  | drive | 121 | 22.3 [8.2, 36.5] | 15.2 [5.5, 26.8] | 33.3 / 40.5 | SMALL 100.0 / MID 6.1 / LARGE 0.0 | 52.8 / 26.1; das 56.0 / 25.5 |
|  | vehicle | 22 | 36.4 [18.2, 54.5] | 17.8 [10.3, 24.8] | 33.3 / 40.9 | SMALL 100.0 / MID 0.0 / LARGE 0.0 | 49.2 / 25.3; das 51.5 / 24.6 |
|  | model | 20 | 40.0 [23.1, 53.8] | 19.0 [12.5, 23.3] | 33.3 / 40.0 | SMALL 100.0 / MID 0.0 / LARGE 0.0 |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Audi A4 Avant | SMALL | 4.409 | yes | HEV | 1 | 43 | 154 | 95.5 | 0.58 | 94.2 | 100.0 | 1/1 | SMALL (0.58, 0.35, 0.07) OK |
| Audi A6 | MID | 4.747 |  | ICE | 1 | 8 | 173 | 100.0 | 0.27 | 0.0 | 0.0 | 0/1 | SMALL (0.66, 0.27, 0.07) MISS |
| Audi e-tron | MID | 4.841 | yes | BEV | 1 | 50 | 219 | 95.4 | 0.35 | 11.9 | 6.0 | 0/1 | SMALL (0.52, 0.35, 0.13) MISS |
| BMW X5 | LARGE | 4.985 |  | ICE | 1 | 8 | 173 | 100.0 | 0.06 | 0.6 | 0.0 | 0/1 | SMALL (0.68, 0.27, 0.06) MISS |
| BMW i3s REx | SMALL | 4.073 |  | PHEV | 1 | 62 | 236 | 96.2 | 0.53 | 83.5 | 93.5 | 1/1 | SMALL (0.53, 0.34, 0.13) OK |
| Ford Fusion | MID | 4.529 |  | unknown | 1 | 1 | 1 | 0.0 | 0.06 | 0.0 | 0.0 | 0/1 | SMALL (0.91, 0.06, 0.03) MISS |
| Ford S-Max | MID | 4.575 |  | ICE | 1 | 7 | 59 | 100.0 | 0.18 | 1.7 | 0.0 | 0/1 | SMALL (0.62, 0.18, 0.20) MISS |
| Hyundai Nexo | MID | 4.517 |  | FCEV | 1 | 9 | 51 | 100.0 | 0.24 | 3.9 | 0.0 | 0/1 | SMALL (0.70, 0.24, 0.06) MISS |
| Jaguar I-Pace | LARGE | 4.917 | yes | BEV | 1 | 46 | 195 | 95.9 | 0.08 | 0.0 | 0.0 | 0/1 | SMALL (0.59, 0.33, 0.08) MISS |
| Kia Niro PHEV | SMALL | 4.293 |  | PHEV | 1 | 7 | 64 | 100.0 | 0.70 | 85.9 | 100.0 | 1/1 | SMALL (0.70, 0.13, 0.17) OK |
| Lincoln MKZ | MID | 4.499 |  | unknown | 2 | 27 | 53 | 100.0 | 0.39 | 18.9 | 7.4 | 0/2 | SMALL (0.46, 0.39, 0.15) MISS |
| Mazda 3 | SMALL | 4.205 |  | ICE | 1 | 20 | 79 | 100.0 | 0.52 | 87.3 | 80.0 | 1/1 | SMALL (0.52, 0.34, 0.14) OK |
| Mercedes-Benz A-Class | SMALL | 4.249 |  | ICE | 1 | 8 | 173 | 100.0 | 0.67 | 98.8 | 100.0 | 1/1 | SMALL (0.67, 0.28, 0.06) OK |
| Mercedes-Benz GLE 450 4MATIC | LARGE | 5.023 |  | HEV | 1 | 66 | 272 | 96.0 | 0.08 | 1.1 | 1.5 | 0/1 | SMALL (0.60, 0.32, 0.08) MISS |
| Mitsubishi Outlander PHEV | SMALL | 4.112 |  | PHEV | 1 | 6 | 53 | 100.0 | 0.64 | 84.9 | 100.0 | 1/1 | SMALL (0.64, 0.14, 0.22) OK |
| Tesla Model 3 | MID | 4.571 |  | BEV | 2 | 53 | 388 | 96.9 | 0.31 | 3.4 | 3.8 | 0/2 | SMALL (0.60, 0.31, 0.09) MISS |
| Tesla Model S | LARGE | 4.973 |  | BEV | 1 | 62 | 255 | 95.7 | 0.08 | 1.2 | 0.0 | 0/1 | SMALL (0.58, 0.34, 0.08) MISS |
| Tesla Model X | LARGE | 5.061 |  | BEV | 1 | 50 | 218 | 98.2 | 0.12 | 4.6 | 4.0 | 0/1 | SMALL (0.55, 0.33, 0.12) MISS |
| Toyota RAV4 Hybrid | SMALL | 4.344 |  | HEV | 1 | 27 | 105 | 100.0 | 0.49 | 76.2 | 77.8 | 1/1 | SMALL (0.49, 0.33, 0.18) OK |
| Volkswagen e-Golf | SMALL | 4.020 |  | BEV | 1 | 6 | 55 | 100.0 | 0.78 | 98.2 | 100.0 | 1/1 | SMALL (0.78, 0.12, 0.10) OK |



Primary run without the borderline models (Audi A4 Avant, Audi e-tron, Jaguar I-Pace):

| level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | support |
|---|---|---|---|---|---|
| window | 2408 | 29.7 [12.1, 55.4] | 18.3 [9.1, 29.0] | SMALL 87.7 / MID 3.6 / LARGE 1.9 | 765/725/918 |
| drive | 121 | 23.1 [9.6, 42.0] | 15.0 [6.9, 22.3] | SMALL 100.0 / MID 4.3 / LARGE 0.0 | 26/46/49 |
| vehicle | 19 | 36.8 [15.8, 57.9] | 17.9 [9.1, 24.4] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | 7/8/4 |
| model | 17 | 41.2 [20.0, 55.6] | 19.4 [11.1, 24.6] | SMALL 100.0 / MID 0.0 / LARGE 0.0 | 7/6/4 |

## openacc: OpenACC platoon campaigns (raw JRC files)

### openacc: bev_ice (BEV / ICE)

Windows 4540 (3955 with a bev_ice label), vehicles 21, drives 630, car models 20; label counts {'BEV': 1880, 'ICE': 2075}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (longitudinal-only, 11 ch) | window | 3955 | 68.2 [54.8, 79.3] | 67.6 [53.3, 77.7] | 50.0 / 52.5 | BEV 86.3 / ICE 51.8 | 76.7 / 74.1; das 75.3 / 73.0 |
|  | drive | 512 | 80.3 [64.2, 90.6] | 76.3 [61.9, 85.6] | 50.0 / 64.5 | BEV 93.9 / ICE 55.5 | 90.0 / 89.0; das 86.9 / 86.3 |
|  | vehicle | 19 | 73.7 [52.6, 94.7] | 73.7 [50.4, 94.2] | 50.0 / 63.2 | BEV 100.0 / ICE 58.3 | 91.4 / 90.4; das 85.4 / 84.2 |
|  | model | 18 | 72.2 [55.6, 90.9] | 72.1 [54.5, 90.9] | 50.0 / 66.7 | BEV 100.0 / ICE 58.3 |  |
| V3 variant (speed-only, 3 ch) | window | 3955 | 65.6 [51.4, 80.9] | 61.5 [50.4, 70.3] | 50.0 / 52.5 | BEV 34.7 / ICE 93.6 | 77.0 / 74.6; das 75.1 / 73.4 |
|  | drive | 512 | 58.4 [42.1, 77.7] | 58.0 [41.0, 73.3] | 50.0 / 64.5 | BEV 37.9 / ICE 95.6 | 89.5 / 88.6; das 88.3 / 87.9 |
|  | vehicle | 19 | 68.4 [47.4, 89.5] | 52.5 [32.1, 73.7] | 50.0 / 63.2 | BEV 14.3 / ICE 100.0 | 91.0 / 90.1; das 86.5 / 85.6 |
|  | model | 18 | 72.2 [55.6, 84.6] | 55.7 [36.8, 70.0] | 50.0 / 66.7 | BEV 16.7 / ICE 100.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 3955 | 50.5 [39.4, 63.4] | 48.5 [39.3, 55.8] | 50.0 / 52.5 | BEV 32.1 / ICE 67.3 | 71.0 / 60.9; das 71.4 / 60.9 |
|  | drive | 512 | 44.1 [29.2, 65.9] | 43.3 [28.0, 56.4] | 50.0 / 64.5 | BEV 24.8 / ICE 79.1 | 67.5 / 54.4; das 66.0 / 53.7 |
|  | vehicle | 19 | 36.8 [15.8, 57.9] | 26.9 [13.6, 36.7] | 50.0 / 63.2 | BEV 0.0 / ICE 58.3 | 69.3 / 56.8; das 66.7 / 51.2 |
|  | model | 18 | 38.9 [21.4, 54.5] | 28.0 [17.6, 35.3] | 50.0 / 66.7 | BEV 0.0 / ICE 58.3 |  |
| full 20-ch model, V3 mask (secondary) | window | 3955 | 52.7 [36.3, 72.0] | 43.9 [34.3, 50.9] | 50.0 / 52.5 | BEV 13.7 / ICE 88.0 | 69.7 / 59.4; das 70.3 / 62.0 |
|  | drive | 512 | 40.6 [21.1, 70.5] | 34.5 [20.5, 48.4] | 50.0 / 64.5 | BEV 7.9 / ICE 100.0 | 63.3 / 51.0; das 69.4 / 60.7 |
|  | vehicle | 19 | 63.2 [42.1, 84.2] | 38.7 [29.6, 45.7] | 50.0 / 63.2 | BEV 0.0 / ICE 100.0 | 65.2 / 49.8; das 70.3 / 59.0 |
|  | model | 18 | 66.7 [50.0, 81.8] | 40.0 [33.3, 45.0] | 50.0 / 66.7 | BEV 0.0 / ICE 100.0 |  |

Per car model, primary run (V2 variant (longitudinal-only, 11 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Audi A6 | ICE | 4.747 |  | ICE | 1 | 10 | 220 | 79.1 | 0.28 | 14.5 | 0.0 | 0/1 | BEV (0.72, 0.28) MISS |
| Audi A8 | ICE | 4.912 | yes | ICE | 1 | 10 | 220 | 100.0 | 0.78 | 90.5 | 100.0 | 1/1 | ICE (0.22, 0.78) OK |
| Audi e-tron | BEV | 4.841 | yes | BEV | 1 | 67 | 350 | 90.3 | 0.79 | 86.9 | 88.1 | 1/1 | BEV (0.79, 0.21) OK |
| BMW X5 | ICE | 4.985 |  | ICE | 1 | 10 | 220 | 79.1 | 0.36 | 24.5 | 10.0 | 0/1 | BEV (0.64, 0.36) MISS |
| Fiat 500X | ICE | 3.971 |  | ICE | 1 | 6 | 182 | 0.0 | 0.70 | 80.2 | 83.3 | 1/1 | ICE (0.30, 0.70) OK |
| Ford S-Max | ICE | 4.575 |  | ICE | 1 | 36 | 217 | 83.1 | 0.57 | 59.4 | 72.2 | 1/1 | ICE (0.43, 0.57) OK |
| Jaguar I-Pace | BEV | 4.917 | yes | BEV | 1 | 60 | 292 | 88.7 | 0.79 | 85.6 | 96.7 | 1/1 | BEV (0.79, 0.21) OK |
| MINI Cooper | ICE | 3.745 |  | ICE | 1 | 5 | 42 | 0.0 | 0.61 | 66.7 | 100.0 | 1/1 | ICE (0.39, 0.61) OK |
| Mazda 3 | ICE | 4.292 |  | ICE | 1 | 63 | 305 | 88.9 | 0.46 | 45.2 | 42.9 | 0/1 | BEV (0.54, 0.46) MISS |
| Mercedes-Benz A-Class | ICE | 4.249 |  | ICE | 1 | 10 | 220 | 79.1 | 0.47 | 51.4 | 30.0 | 0/1 | BEV (0.53, 0.47) MISS |
| Mitsubishi Space Star | ICE | 3.485 |  | ICE | 1 | 12 | 114 | 0.0 | 0.71 | 81.6 | 100.0 | 1/1 | ICE (0.28, 0.71) OK |
| SsangYong Rexton | ICE | 4.699 |  | ICE | 1 | 9 | 69 | 0.0 | 0.62 | 68.1 | 66.7 | 1/1 | ICE (0.38, 0.62) OK |
| Tesla Model 3 | BEV | 4.571 |  | BEV | 2 | 58 | 486 | 84.4 | 0.79 | 91.8 | 96.6 | 2/2 | BEV (0.79, 0.21) OK |
| Tesla Model S | BEV | 4.973 |  | BEV | 1 | 69 | 334 | 89.8 | 0.79 | 90.1 | 98.6 | 1/1 | BEV (0.79, 0.21) OK |
| Tesla Model X | BEV | 5.061 |  | BEV | 1 | 69 | 345 | 90.4 | 0.73 | 83.5 | 94.2 | 1/1 | BEV (0.73, 0.27) OK |
| Volkswagen Polo | ICE | 3.605 |  | ICE | 1 | 5 | 82 | 0.0 | 0.70 | 78.0 | 100.0 | 1/1 | ICE (0.30, 0.70) OK |
| Volkswagen e-Golf | BEV | 4.020 |  | BEV | 1 | 7 | 73 | 90.1 | 0.51 | 46.6 | 57.1 | 1/1 | BEV (0.51, 0.49) OK |
| Volvo XC40 | ICE | 4.360 |  | ICE | 1 | 6 | 184 | 86.5 | 0.27 | 16.8 | 16.7 | 0/1 | BEV (0.73, 0.27) MISS |

Windows outside the bev_ice classes (scored, not in the metrics), primary run:

| model | label | veh | win | predicted share % | mean P |
|---|---|---|---|---|---|
| Audi A4 Avant | HEV | 1 | 225 | BEV 82.7, ICE 17.3 | 0.71, 0.29 |
| Mercedes-Benz GLE 450 4MATIC | HEV | 1 | 360 | BEV 49.7, ICE 50.3 | 0.49, 0.51 |

### openacc: size (SMALL / MID / LARGE)

Windows 4540 (4540 with a size label), vehicles 21, drives 630, car models 20; label counts {'SMALL': 1427, 'MID': 1342, 'LARGE': 1771}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (longitudinal-only, 11 ch) | window | 4540 | 29.6 [20.0, 39.7] | 28.6 [19.1, 35.1] | 33.3 / 39.0 | SMALL 37.6 / MID 46.6 / LARGE 10.3 | 41.7 / 34.9; das 43.0 / 35.0 |
|  | drive | 630 | 25.1 [12.4, 39.4] | 23.0 [13.1, 30.7] | 33.3 / 45.9 | SMALL 47.2 / MID 42.2 / LARGE 2.1 | 45.2 / 33.5; das 46.8 / 34.3 |
|  | vehicle | 21 | 42.9 [23.8, 61.9] | 36.4 [19.0, 49.2] | 33.3 / 42.9 | SMALL 55.6 / MID 66.7 / LARGE 0.0 | 43.2 / 32.8; das 46.2 / 36.0 |
|  | model | 20 | 40.0 [25.0, 58.3] | 34.1 [20.6, 46.3] | 33.3 / 45.0 | SMALL 55.6 / MID 60.0 / LARGE 0.0 |  |
| V3 variant (speed-only, 3 ch) | window | 4540 | 30.2 [15.8, 47.0] | 22.0 [14.0, 28.6] | 33.3 / 39.0 | SMALL 82.1 / MID 10.1 / LARGE 3.6 | 47.1 / 41.3; das 47.3 / 40.1 |
|  | drive | 630 | 27.1 [10.0, 50.8] | 19.9 [9.7, 28.2] | 33.3 / 45.9 | SMALL 92.5 / MID 9.4 / LARGE 1.7 | 53.4 / 45.1; das 49.4 / 39.3 |
|  | vehicle | 21 | 38.1 [19.0, 57.1] | 18.4 [10.7, 24.2] | 33.3 / 42.9 | SMALL 88.9 / MID 0.0 / LARGE 0.0 | 54.4 / 45.6; das 56.1 / 47.0 |
|  | model | 20 | 40.0 [23.1, 54.5] | 19.0 [12.5, 23.5] | 33.3 / 45.0 | SMALL 88.9 / MID 0.0 / LARGE 0.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 4540 | 27.8 [13.5, 44.7] | 23.4 [13.5, 32.5] | 33.3 / 39.0 | SMALL 66.9 / MID 11.5 / LARGE 8.8 | 53.6 / 33.2; das 57.2 / 37.9 |
|  | drive | 630 | 24.3 [6.0, 47.5] | 17.3 [5.8, 27.9] | 33.3 / 45.9 | SMALL 82.6 / MID 3.3 / LARGE 4.8 | 52.7 / 24.8; das 56.6 / 28.3 |
|  | vehicle | 21 | 23.8 [9.4, 42.9] | 13.3 [5.7, 20.7] | 33.3 / 42.9 | SMALL 55.6 / MID 0.0 / LARGE 0.0 | 48.0 / 23.2; das 51.1 / 26.3 |
|  | model | 20 | 25.0 [8.3, 38.5] | 13.3 [5.1, 19.0] | 33.3 / 45.0 | SMALL 55.6 / MID 0.0 / LARGE 0.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 4540 | 31.3 [13.7, 53.0] | 18.6 [10.3, 26.8] | 33.3 / 39.0 | SMALL 94.9 / MID 3.9 / LARGE 0.9 | 55.1 / 30.1; das 59.4 / 30.7 |
|  | drive | 630 | 26.2 [6.8, 52.9] | 15.0 [5.1, 25.8] | 33.3 / 45.9 | SMALL 100.0 / MID 1.7 / LARGE 0.3 | 52.8 / 26.1; das 56.0 / 25.5 |
|  | vehicle | 21 | 42.9 [23.8, 66.7] | 20.0 [12.8, 26.7] | 33.3 / 42.9 | SMALL 100.0 / MID 0.0 / LARGE 0.0 | 49.2 / 25.3; das 51.5 / 24.6 |
|  | model | 20 | 45.0 [27.3, 61.5] | 20.7 [14.3, 25.4] | 33.3 / 45.0 | SMALL 100.0 / MID 0.0 / LARGE 0.0 |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Audi A4 Avant | SMALL | 4.409 | yes | HEV | 1 | 47 | 225 | 92.4 | 0.56 | 88.4 | 97.9 | 1/1 | SMALL (0.56, 0.36, 0.08) OK |
| Audi A6 | MID | 4.747 |  | ICE | 1 | 10 | 220 | 79.1 | 0.27 | 3.2 | 0.0 | 0/1 | SMALL (0.65, 0.27, 0.08) MISS |
| Audi A8 | LARGE | 4.912 | yes | ICE | 1 | 10 | 220 | 100.0 | 0.12 | 8.2 | 0.0 | 0/1 | SMALL (0.64, 0.25, 0.12) MISS |
| Audi e-tron | MID | 4.841 | yes | BEV | 1 | 67 | 350 | 90.3 | 0.36 | 15.7 | 13.4 | 0/1 | SMALL (0.51, 0.35, 0.13) MISS |
| BMW X5 | LARGE | 4.985 |  | ICE | 1 | 10 | 220 | 79.1 | 0.07 | 0.5 | 0.0 | 0/1 | SMALL (0.66, 0.27, 0.07) MISS |
| Fiat 500X | SMALL | 3.971 |  | ICE | 1 | 6 | 182 | 0.0 | 0.64 | 89.0 | 83.3 | 1/1 | SMALL (0.64, 0.16, 0.20) OK |
| Ford S-Max | MID | 4.575 |  | ICE | 1 | 36 | 217 | 83.1 | 0.27 | 12.0 | 11.1 | 0/1 | SMALL (0.53, 0.27, 0.20) MISS |
| Jaguar I-Pace | LARGE | 4.917 | yes | BEV | 1 | 60 | 292 | 88.7 | 0.09 | 1.4 | 0.0 | 0/1 | SMALL (0.57, 0.33, 0.09) MISS |
| MINI Cooper | SMALL | 3.745 |  | ICE | 1 | 5 | 42 | 0.0 | 0.52 | 78.6 | 100.0 | 1/1 | SMALL (0.52, 0.23, 0.25) OK |
| Mazda 3 | SMALL | 4.292 |  | ICE | 1 | 63 | 305 | 88.9 | 0.51 | 81.0 | 93.7 | 1/1 | SMALL (0.51, 0.34, 0.15) OK |
| Mercedes-Benz A-Class | SMALL | 4.249 |  | ICE | 1 | 10 | 220 | 79.1 | 0.65 | 95.9 | 100.0 | 1/1 | SMALL (0.65, 0.28, 0.07) OK |
| Mercedes-Benz GLE 450 4MATIC | LARGE | 5.023 |  | HEV | 1 | 71 | 360 | 91.1 | 0.08 | 1.7 | 0.0 | 0/1 | SMALL (0.59, 0.33, 0.08) MISS |
| Mitsubishi Space Star | SMALL | 3.485 |  | ICE | 1 | 12 | 114 | 0.0 | 0.64 | 83.3 | 100.0 | 1/1 | SMALL (0.64, 0.17, 0.20) OK |
| SsangYong Rexton | MID | 4.699 |  | ICE | 1 | 9 | 69 | 0.0 | 0.31 | 21.7 | 0.0 | 0/1 | SMALL (0.62, 0.31, 0.07) MISS |
| Tesla Model 3 | MID | 4.571 |  | BEV | 2 | 58 | 486 | 84.4 | 0.32 | 6.8 | 6.9 | 0/2 | SMALL (0.59, 0.32, 0.09) MISS |
| Tesla Model S | LARGE | 4.973 |  | BEV | 1 | 69 | 334 | 89.8 | 0.10 | 3.0 | 1.4 | 0/1 | SMALL (0.55, 0.35, 0.10) MISS |
| Tesla Model X | LARGE | 5.061 |  | BEV | 1 | 69 | 345 | 90.4 | 0.12 | 7.0 | 5.8 | 0/1 | SMALL (0.53, 0.35, 0.12) MISS |
| Volkswagen Polo | SMALL | 3.605 |  | ICE | 1 | 5 | 82 | 0.0 | 0.25 | 8.5 | 0.0 | 0/1 | LARGE (0.25, 0.30, 0.45) MISS |
| Volkswagen e-Golf | SMALL | 4.020 |  | BEV | 1 | 7 | 73 | 90.1 | 0.71 | 89.0 | 100.0 | 1/1 | SMALL (0.71, 0.17, 0.12) OK |
| Volvo XC40 | SMALL | 4.360 |  | ICE | 1 | 6 | 184 | 86.5 | 0.72 | 82.6 | 83.3 | 1/1 | SMALL (0.72, 0.14, 0.14) OK |



Primary run without the borderline models (Audi A4 Avant, Audi A8, Audi e-tron, Jaguar I-Pace):

| level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | support |
|---|---|---|---|---|---|
| window | 3453 | 31.7 [14.5, 52.2] | 21.7 [12.3, 29.8] | SMALL 80.9 / MID 8.2 / LARGE 3.3 | 1202/992/1259 |
| drive | 446 | 26.0 [8.1, 54.5] | 18.5 [7.8, 29.8] | SMALL 90.4 / MID 7.1 / LARGE 2.3 | 114/113/219 |
| vehicle | 17 | 41.2 [17.6, 64.7] | 19.4 [10.0, 26.2] | SMALL 87.5 / MID 0.0 / LARGE 0.0 | 8/5/4 |
| model | 16 | 43.8 [22.2, 60.0] | 20.3 [12.1, 25.9] | SMALL 87.5 / MID 0.0 / LARGE 0.0 | 8/4/4 |

## microsim_ev: microSIM-ACC / EV

### microsim_ev: bev_ice (BEV / ICE)

Windows 111 (111 with a bev_ice label), vehicles 3, drives 110, car models 3; label counts {'BEV': 111, 'ICE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (longitudinal-only, 11 ch) | window | 111 | 55.0 [50.0, 100.0] | - | 50.0 / 100.0 | BEV 55.0 / ICE - | 76.7 / 74.1; das 75.3 / 73.0 |
|  | drive | 110 | 54.5 [50.0, 100.0] | - | 50.0 / 100.0 | BEV 54.5 / ICE - | 90.0 / 89.0; das 86.9 / 86.3 |
|  | vehicle | 3 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV 100.0 / ICE - | 91.4 / 90.4; das 85.4 / 84.2 |
|  | model | 3 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV 100.0 / ICE - |  |
| V3 variant (speed-only, 3 ch) | window | 111 | 11.7 [11.1, 25.0] | - | 50.0 / 100.0 | BEV 11.7 / ICE - | 77.0 / 74.6; das 75.1 / 73.4 |
|  | drive | 110 | 11.8 [11.2, 25.0] | - | 50.0 / 100.0 | BEV 11.8 / ICE - | 89.5 / 88.6; das 88.3 / 87.9 |
|  | vehicle | 3 | 0.0 [0.0, 0.0] | - | 50.0 / 100.0 | BEV 0.0 / ICE - | 91.0 / 90.1; das 86.5 / 85.6 |
|  | model | 3 | 0.0 [0.0, 0.0] | - | 50.0 / 100.0 | BEV 0.0 / ICE - |  |
| full 20-ch model, V2 mask (secondary) | window | 111 | 61.3 [0.0, 63.6] | - | 50.0 / 100.0 | BEV 61.3 / ICE - | 71.0 / 60.9; das 71.4 / 60.9 |
|  | drive | 110 | 60.9 [0.0, 63.3] | - | 50.0 / 100.0 | BEV 60.9 / ICE - | 67.5 / 54.4; das 66.0 / 53.7 |
|  | vehicle | 3 | 66.7 [0.0, 100.0] | - | 50.0 / 100.0 | BEV 66.7 / ICE - | 69.3 / 56.8; das 66.7 / 51.2 |
|  | model | 3 | 66.7 [0.0, 100.0] | - | 50.0 / 100.0 | BEV 66.7 / ICE - |  |
| full 20-ch model, V3 mask (secondary) | window | 111 | 18.0 [14.1, 62.5] | - | 50.0 / 100.0 | BEV 18.0 / ICE - | 69.7 / 59.4; das 70.3 / 62.0 |
|  | drive | 110 | 18.2 [14.3, 62.5] | - | 50.0 / 100.0 | BEV 18.2 / ICE - | 63.3 / 51.0; das 69.4 / 60.7 |
|  | vehicle | 3 | 33.3 [0.0, 100.0] | - | 50.0 / 100.0 | BEV 33.3 / ICE - | 65.2 / 49.8; das 70.3 / 59.0 |
|  | model | 3 | 33.3 [0.0, 100.0] | - | 50.0 / 100.0 | BEV 33.3 / ICE - |  |

Per car model, primary run (V2 variant (longitudinal-only, 11 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hyundai Ioniq 5 | BEV | 4.826 | yes | BEV | 1 | 98 | 99 | 100.0 | 0.54 | 51.5 | 51.0 | 1/1 | BEV (0.54, 0.46) OK |
| Polestar 2 | BEV | 4.380 | yes | BEV | 1 | 8 | 8 | 100.0 | 0.92 | 100.0 | 100.0 | 1/1 | BEV (0.92, 0.08) OK |
| Tesla Model 3 | BEV | 4.571 |  | BEV | 1 | 4 | 4 | 100.0 | 0.54 | 50.0 | 50.0 | 1/1 | BEV (0.54, 0.46) OK |

### microsim_ev: size (SMALL / MID / LARGE)

Windows 111 (111 with a size label), vehicles 3, drives 110, car models 3; label counts {'SMALL': 8, 'MID': 103, 'LARGE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (longitudinal-only, 11 ch) | window | 111 | 25.2 [0.0, 27.3] | 20.1 [4.8, 20.8] | 33.3 / 92.8 | SMALL 0.0 / MID 27.2 / LARGE - | 41.7 / 34.9; das 43.0 / 35.0 |
|  | drive | 110 | 24.5 [0.0, 26.5] | 19.7 [4.8, 20.3] | 33.3 / 92.7 | SMALL 0.0 / MID 26.5 / LARGE - | 45.2 / 33.5; das 46.8 / 34.3 |
|  | vehicle | 3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 33.3 / 66.7 | SMALL 0.0 / MID 0.0 / LARGE - | 43.2 / 32.8; das 46.2 / 36.0 |
|  | model | 3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 33.3 / 66.7 | SMALL 0.0 / MID 0.0 / LARGE - |  |
| V3 variant (speed-only, 3 ch) | window | 111 | 26.1 [0.0, 100.0] | 25.7 [22.6, 44.4] | 33.3 / 92.8 | SMALL 100.0 / MID 20.4 / LARGE - | 47.1 / 41.3; das 47.3 / 40.1 |
|  | drive | 110 | 26.4 [0.0, 100.0] | 26.0 [22.8, 44.4] | 33.3 / 92.7 | SMALL 100.0 / MID 20.6 / LARGE - | 53.4 / 45.1; das 49.4 / 39.3 |
|  | vehicle | 3 | 33.3 [0.0, 100.0] | 25.0 [25.0, 40.0] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - | 54.4 / 45.6; das 56.1 / 47.0 |
|  | model | 3 | 33.3 [0.0, 100.0] | 25.0 [25.0, 33.3] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - |  |
| full 20-ch model, V2 mask (secondary) | window | 111 | 23.4 [0.0, 25.0] | 21.6 [12.5, 24.5] | 33.3 / 92.8 | SMALL 25.0 / MID 23.3 / LARGE - | 53.6 / 33.2; das 57.2 / 37.9 |
|  | drive | 110 | 23.6 [0.0, 25.0] | 21.8 [12.5, 24.7] | 33.3 / 92.7 | SMALL 25.0 / MID 23.5 / LARGE - | 52.7 / 24.8; das 56.6 / 28.3 |
|  | vehicle | 3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 33.3 / 66.7 | SMALL 0.0 / MID 0.0 / LARGE - | 48.0 / 23.2; das 51.1 / 26.3 |
|  | model | 3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 33.3 / 66.7 | SMALL 0.0 / MID 0.0 / LARGE - |  |
| full 20-ch model, V3 mask (secondary) | window | 111 | 9.9 [0.0, 100.0] | 9.7 [6.8, 44.4] | 33.3 / 92.8 | SMALL 100.0 / MID 2.9 / LARGE - | 55.1 / 30.1; das 59.4 / 30.7 |
|  | drive | 110 | 10.0 [0.0, 100.0] | 9.8 [6.9, 44.4] | 33.3 / 92.7 | SMALL 100.0 / MID 2.9 / LARGE - | 52.8 / 26.1; das 56.0 / 25.5 |
|  | vehicle | 3 | 33.3 [0.0, 100.0] | 25.0 [25.0, 40.0] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - | 49.2 / 25.3; das 51.5 / 24.6 |
|  | model | 3 | 33.3 [0.0, 100.0] | 25.0 [25.0, 33.3] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hyundai Ioniq 5 | MID | 4.826 | yes | BEV | 1 | 98 | 99 | 100.0 | 0.38 | 21.2 | 21.4 | 0/1 | SMALL (0.48, 0.38, 0.14) MISS |
| Polestar 2 | SMALL | 4.380 | yes | BEV | 1 | 8 | 8 | 100.0 | 0.60 | 100.0 | 100.0 | 1/1 | SMALL (0.60, 0.35, 0.04) OK |
| Tesla Model 3 | MID | 4.571 |  | BEV | 1 | 4 | 4 | 100.0 | 0.41 | 0.0 | 0.0 | 0/1 | SMALL (0.47, 0.41, 0.12) MISS |



Primary run without the borderline models (Hyundai Ioniq 5, Polestar 2):

| level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | support |
|---|---|---|---|---|---|
| window | 4 | 0.0 [0.0, 0.0] | - | SMALL - / MID 0.0 / LARGE - | 0/4/0 |
| drive | 4 | 0.0 [0.0, 0.0] | - | SMALL - / MID 0.0 / LARGE - | 0/4/0 |
| vehicle | 1 | 0.0 [0.0, 0.0] | - | SMALL - / MID 0.0 / LARGE - | 0/1/0 |
| model | 1 | 0.0 [0.0, 0.0] | - | SMALL - / MID 0.0 / LARGE - | 0/1/0 |

## uga_ev: UGA EV platoon

### uga_ev: bev_ice (BEV / ICE)

Windows 326 (326 with a bev_ice label), vehicles 4, drives 48, car models 3; label counts {'BEV': 244, 'ICE': 82}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (longitudinal-only, 11 ch) | window | 326 | 42.6 [26.1, 59.6] | 37.6 [25.4, 44.4] | 50.0 / 74.8 | BEV 47.5 / ICE 28.0 | 76.7 / 74.1; das 75.3 / 73.0 |
|  | drive | 48 | 47.9 [25.0, 70.8] | 42.3 [24.5, 54.0] | 50.0 / 75.0 | BEV 52.8 / ICE 33.3 | 90.0 / 89.0; das 86.9 / 86.3 |
|  | vehicle | 4 | 50.0 [0.0, 100.0] | 33.3 [0.0, 42.9] | 50.0 / 75.0 | BEV 66.7 / ICE 0.0 | 91.4 / 90.4; das 85.4 / 84.2 |
|  | model | 3 | 33.3 [0.0, 100.0] | 25.0 [0.0, 33.3] | 50.0 / 66.7 | BEV 50.0 / ICE 0.0 |  |
| V3 variant (speed-only, 3 ch) | window | 326 | 34.0 [9.3, 75.1] | 32.8 [27.7, 59.1] | 50.0 / 74.8 | BEV 13.5 / ICE 95.1 | 77.0 / 74.6; das 75.1 / 73.4 |
|  | drive | 48 | 25.0 [0.0, 75.0] | 20.0 [20.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 | 89.5 / 88.6; das 88.3 / 87.9 |
|  | vehicle | 4 | 25.0 [0.0, 75.0] | 20.0 [20.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 | 91.0 / 90.1; das 86.5 / 85.6 |
|  | model | 3 | 33.3 [0.0, 50.0] | 25.0 [25.0, 33.3] | 50.0 / 66.7 | BEV 0.0 / ICE 100.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 326 | 34.7 [12.7, 66.2] | 34.4 [28.1, 55.0] | 50.0 / 74.8 | BEV 19.3 / ICE 80.5 | 71.0 / 60.9; das 71.4 / 60.9 |
|  | drive | 48 | 27.1 [4.2, 64.6] | 25.5 [20.1, 44.2] | 50.0 / 75.0 | BEV 8.3 / ICE 83.3 | 67.5 / 54.4; das 66.0 / 53.7 |
|  | vehicle | 4 | 25.0 [0.0, 75.0] | 20.0 [20.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 | 69.3 / 56.8; das 66.7 / 51.2 |
|  | model | 3 | 33.3 [0.0, 50.0] | 25.0 [25.0, 33.3] | 50.0 / 66.7 | BEV 0.0 / ICE 100.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 326 | 23.3 [0.9, 65.7] | 20.0 [17.8, 41.5] | 50.0 / 74.8 | BEV 2.0 / ICE 86.6 | 69.7 / 59.4; das 70.3 / 62.0 |
|  | drive | 48 | 25.0 [0.0, 75.0] | 20.0 [20.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 | 63.3 / 51.0; das 69.4 / 60.7 |
|  | vehicle | 4 | 25.0 [0.0, 75.0] | 20.0 [20.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 | 65.2 / 49.8; das 70.3 / 59.0 |
|  | model | 3 | 33.3 [0.0, 50.0] | 25.0 [25.0, 33.3] | 50.0 / 66.7 | BEV 0.0 / ICE 100.0 |  |

Per car model, primary run (V2 variant (longitudinal-only, 11 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ford Mustang Mach-E | BEV | 4.845 | yes | BEV | 1 | 12 | 83 | 67.5 | 0.36 | 24.1 | 16.7 | 0/1 | ICE (0.36, 0.64) MISS |
| Kia Niro EV | BEV | 4.255 |  | BEV | 2 | 24 | 161 | 67.1 | 0.55 | 59.6 | 70.8 | 2/2 | BEV (0.55, 0.45) OK |
| Lexus UX 200 | ICE | 4.092 |  | ICE | 1 | 12 | 82 | 0.0 | 0.42 | 28.0 | 33.3 | 0/1 | BEV (0.58, 0.42) MISS |

### uga_ev: size (SMALL / MID / LARGE)

Windows 326 (326 with a size label), vehicles 4, drives 48, car models 3; label counts {'SMALL': 243, 'MID': 83, 'LARGE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V2 variant (longitudinal-only, 11 ch) | window | 326 | 39.0 [23.0, 54.5] | 45.5 [35.2, 66.8] | 33.3 / 74.5 | SMALL 34.6 / MID 51.8 / LARGE - | 41.7 / 34.9; das 43.0 / 35.0 |
|  | drive | 48 | 43.8 [20.8, 66.7] | 47.7 [31.8, 76.4] | 33.3 / 75.0 | SMALL 36.1 / MID 66.7 / LARGE - | 45.2 / 33.5; das 46.8 / 34.3 |
|  | vehicle | 4 | 50.0 [0.0, 100.0] | 50.0 [20.0, 100.0] | 33.3 / 75.0 | SMALL 33.3 / MID 100.0 / LARGE - | 43.2 / 32.8; das 46.2 / 36.0 |
|  | model | 3 | 66.7 [0.0, 100.0] | 66.7 [33.3, 100.0] | 33.3 / 66.7 | SMALL 50.0 / MID 100.0 / LARGE - |  |
| V3 variant (speed-only, 3 ch) | window | 326 | 73.9 [39.9, 98.2] | 56.7 [39.2, 60.5] | 33.3 / 74.5 | SMALL 92.2 / MID 20.5 / LARGE - | 47.1 / 41.3; das 47.3 / 40.1 |
|  | drive | 48 | 75.0 [25.0, 100.0] | 42.9 [20.0, 42.9] | 33.3 / 75.0 | SMALL 100.0 / MID 0.0 / LARGE - | 53.4 / 45.1; das 49.4 / 39.3 |
|  | vehicle | 4 | 75.0 [25.0, 100.0] | 42.9 [20.0, 42.9] | 33.3 / 75.0 | SMALL 100.0 / MID 0.0 / LARGE - | 54.4 / 45.6; das 56.1 / 47.0 |
|  | model | 3 | 66.7 [50.0, 100.0] | 40.0 [33.3, 40.0] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - |  |
| full 20-ch model, V2 mask (secondary) | window | 326 | 59.2 [22.3, 90.2] | 45.7 [25.6, 49.8] | 33.3 / 74.5 | SMALL 77.0 / MID 7.2 / LARGE - | 53.6 / 33.2; das 57.2 / 37.9 |
|  | drive | 48 | 56.2 [16.7, 89.6] | 39.1 [21.1, 43.2] | 33.3 / 75.0 | SMALL 75.0 / MID 0.0 / LARGE - | 52.7 / 24.8; das 56.6 / 28.3 |
|  | vehicle | 4 | 75.0 [25.0, 100.0] | 42.9 [20.0, 42.9] | 33.3 / 75.0 | SMALL 100.0 / MID 0.0 / LARGE - | 48.0 / 23.2; das 51.1 / 26.3 |
|  | model | 3 | 66.7 [50.0, 100.0] | 40.0 [33.3, 40.0] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - |  |
| full 20-ch model, V3 mask (secondary) | window | 326 | 72.4 [23.6, 98.1] | 42.4 [19.8, 42.7] | 33.3 / 74.5 | SMALL 97.1 / MID 0.0 / LARGE - | 55.1 / 30.1; das 59.4 / 30.7 |
|  | drive | 48 | 75.0 [25.0, 100.0] | 42.9 [20.0, 42.9] | 33.3 / 75.0 | SMALL 100.0 / MID 0.0 / LARGE - | 52.8 / 26.1; das 56.0 / 25.5 |
|  | vehicle | 4 | 75.0 [25.0, 100.0] | 42.9 [20.0, 42.9] | 33.3 / 75.0 | SMALL 100.0 / MID 0.0 / LARGE - | 49.2 / 25.3; das 51.5 / 24.6 |
|  | model | 3 | 66.7 [50.0, 100.0] | 40.0 [33.3, 40.0] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ford Mustang Mach-E | MID | 4.845 | yes | BEV | 1 | 12 | 83 | 67.5 | 0.37 | 20.5 | 0.0 | 0/1 | SMALL (0.55, 0.37, 0.08) MISS |
| Kia Niro EV | SMALL | 4.255 |  | BEV | 2 | 24 | 161 | 67.1 | 0.61 | 89.4 | 100.0 | 2/2 | SMALL (0.61, 0.30, 0.09) OK |
| Lexus UX 200 | SMALL | 4.092 |  | ICE | 1 | 12 | 82 | 0.0 | 0.60 | 97.6 | 100.0 | 1/1 | SMALL (0.60, 0.34, 0.06) OK |



Primary run without the borderline models (Ford Mustang Mach-E):

| level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | support |
|---|---|---|---|---|---|
| window | 243 | 92.2 [79.7, 98.8] | - | SMALL 92.2 / MID - / LARGE - | 243/0/0 |
| drive | 36 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | 36/0/0 |
| vehicle | 3 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | 3/0/0 |
| model | 2 | 100.0 [100.0, 100.0] | - | SMALL 100.0 / MID - / LARGE - | 2/0/0 |

## tesla_yuan: Yuan et al. 2024, Tesla Model 3 trips

### tesla_yuan: bev_ice (BEV / ICE)

Windows 2619 (2619 with a bev_ice label), vehicles 1, drives 106, car models 1; label counts {'BEV': 2619, 'ICE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `human` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; human) |
|---|---|---|---|---|---|---|---|
| V3 variant (speed-only, 3 ch) | window | 2619 | 62.3 [62.3, 62.3] | - | 50.0 / 100.0 | BEV 62.3 / ICE - | 77.0 / 74.6; human 78.0 / 73.0 |
|  | drive | 106 | 80.2 [80.2, 80.2] | - | 50.0 / 100.0 | BEV 80.2 / ICE - | 89.5 / 88.6; human 87.6 / 85.9 |
|  | vehicle | 1 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV 100.0 / ICE - | 91.0 / 90.1; human 89.8 / 88.6 |
|  | model | 1 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV 100.0 / ICE - |  |
| full 20-ch model, V3 mask (secondary) | window | 2619 | 10.9 [10.9, 10.9] | - | 50.0 / 100.0 | BEV 10.9 / ICE - | 69.7 / 59.4; human 69.8 / 54.5 |
|  | drive | 106 | 0.0 [0.0, 0.0] | - | 50.0 / 100.0 | BEV 0.0 / ICE - | 63.3 / 51.0; human 64.0 / 49.4 |
|  | vehicle | 1 | 0.0 [0.0, 0.0] | - | 50.0 / 100.0 | BEV 0.0 / ICE - | 65.2 / 49.8; human 63.6 / 50.1 |
|  | model | 1 | 0.0 [0.0, 0.0] | - | 50.0 / 100.0 | BEV 0.0 / ICE - |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tesla Model 3 | BEV | 4.571 |  | BEV | 1 | 106 | 2619 | 0.0 | 0.61 | 62.3 | 80.2 | 1/1 | BEV (0.61, 0.39) OK |

### tesla_yuan: size (SMALL / MID / LARGE)

Windows 2619 (2619 with a size label), vehicles 1, drives 106, car models 1; label counts {'SMALL': 0, 'MID': 2619, 'LARGE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `human` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; human) |
|---|---|---|---|---|---|---|---|
| V3 variant (speed-only, 3 ch) | window | 2619 | 63.7 [63.7, 63.7] | - | 33.3 / 100.0 | SMALL - / MID 63.7 / LARGE - | 47.1 / 41.3; human 43.9 / 41.7 |
|  | drive | 106 | 78.3 [78.3, 78.3] | - | 33.3 / 100.0 | SMALL - / MID 78.3 / LARGE - | 53.4 / 45.1; human 50.4 / 46.5 |
|  | vehicle | 1 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL - / MID 100.0 / LARGE - | 54.4 / 45.6; human 48.3 / 44.1 |
|  | model | 1 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL - / MID 100.0 / LARGE - |  |
| full 20-ch model, V3 mask (secondary) | window | 2619 | 20.0 [20.0, 20.0] | - | 33.3 / 100.0 | SMALL - / MID 20.0 / LARGE - | 55.1 / 30.1; human 42.3 / 25.0 |
|  | drive | 106 | 7.5 [7.5, 7.5] | - | 33.3 / 100.0 | SMALL - / MID 7.5 / LARGE - | 52.8 / 26.1; human 47.1 / 26.1 |
|  | vehicle | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID 0.0 / LARGE - | 49.2 / 25.3; human 46.8 / 24.7 |
|  | model | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID 0.0 / LARGE - |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tesla Model 3 | MID | 4.571 |  | BEV | 1 | 106 | 2619 | 0.0 | 0.50 | 63.7 | 78.3 | 1/1 | MID (0.19, 0.50, 0.31) OK |

## openacc_500x: OpenACC one-vehicle campaign (Fiat 500X, 20 drivers)

### openacc_500x: bev_ice (BEV / ICE)

Windows 13523 (13523 with a bev_ice label), vehicles 1, drives 429, car models 1; label counts {'BEV': 0, 'ICE': 13523}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `human` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; human) |
|---|---|---|---|---|---|---|---|
| V3 variant (speed-only, 3 ch) | window | 13523 | 87.9 [87.9, 87.9] | - | 50.0 / 100.0 | BEV - / ICE 87.9 | 77.0 / 74.6; human 78.0 / 73.0 |
|  | drive | 429 | 99.1 [99.1, 99.1] | - | 50.0 / 100.0 | BEV - / ICE 99.1 | 89.5 / 88.6; human 87.6 / 85.9 |
|  | vehicle | 1 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 91.0 / 90.1; human 89.8 / 88.6 |
|  | model | 1 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 13523 | 90.4 [90.4, 90.4] | - | 50.0 / 100.0 | BEV - / ICE 90.4 | 69.7 / 59.4; human 69.8 / 54.5 |
|  | drive | 429 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 63.3 / 51.0; human 64.0 / 49.4 |
|  | vehicle | 1 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 65.2 / 49.8; human 63.6 / 50.1 |
|  | model | 1 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fiat 500X | ICE | 3.971 |  | ICE | 1 | 429 | 13523 | 0.0 | 0.82 | 87.9 | 99.1 | 1/1 | ICE (0.18, 0.82) OK |

### openacc_500x: size (SMALL / MID / LARGE)

Windows 13523 (13523 with a size label), vehicles 1, drives 429, car models 1; label counts {'SMALL': 13523, 'MID': 0, 'LARGE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `human` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; human) |
|---|---|---|---|---|---|---|---|
| V3 variant (speed-only, 3 ch) | window | 13523 | 57.3 [57.3, 57.3] | - | 33.3 / 100.0 | SMALL 57.3 / MID - / LARGE - | 47.1 / 41.3; human 43.9 / 41.7 |
|  | drive | 429 | 46.2 [46.2, 46.2] | - | 33.3 / 100.0 | SMALL 46.2 / MID - / LARGE - | 53.4 / 45.1; human 50.4 / 46.5 |
|  | vehicle | 1 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 54.4 / 45.6; human 48.3 / 44.1 |
|  | model | 1 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, V3 mask (secondary) | window | 13523 | 92.2 [92.2, 92.2] | - | 33.3 / 100.0 | SMALL 92.2 / MID - / LARGE - | 55.1 / 30.1; human 42.3 / 25.0 |
|  | drive | 429 | 98.4 [98.4, 98.4] | - | 33.3 / 100.0 | SMALL 98.4 / MID - / LARGE - | 52.8 / 26.1; human 47.1 / 26.1 |
|  | vehicle | 1 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 49.2 / 25.3; human 46.8 / 24.7 |
|  | model | 1 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fiat 500X | SMALL | 3.971 |  | ICE | 1 | 429 | 13523 | 0.0 | 0.47 | 57.3 | 46.2 | 1/1 | SMALL (0.47, 0.39, 0.14) OK |

## ved: VED, Vehicle Energy Dataset

### ved: bev_ice (BEV / ICE)

Windows 59563 (39259 with a bev_ice label), vehicles 384, drives 25580, car models 21; label counts {'BEV': 431, 'ICE': 38828}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `human` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; human) |
|---|---|---|---|---|---|---|---|
| V3 variant (speed-only, 3 ch) | window | 39259 | 68.8 [67.4, 70.1] | 41.3 [40.7, 42.0] | 50.0 / 98.9 | BEV 17.4 / ICE 69.3 | 77.0 / 74.6; human 78.0 / 73.0 |
|  | drive | 15770 | 74.6 [72.3, 76.6] | 44.0 [42.9, 45.3] | 50.0 / 98.0 | BEV 16.4 / ICE 75.8 | 89.5 / 88.6; human 87.6 / 85.9 |
|  | vehicle | 267 | 96.6 [94.4, 98.5] | 49.1 [48.6, 49.6] | 50.0 / 98.9 | BEV 0.0 / ICE 97.7 | 91.0 / 90.1; human 89.8 / 88.6 |
|  | model | 11 | 90.9 [88.9, 100.0] | 47.6 [47.1, 47.6] | 50.0 / 90.9 | BEV 0.0 / ICE 100.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 39259 | 95.1 [93.7, 96.2] | 50.3 [49.0, 51.6] | 50.0 / 98.9 | BEV 7.2 / ICE 96.1 | 69.7 / 59.4; human 69.8 / 54.5 |
|  | drive | 15770 | 96.1 [93.4, 98.0] | 51.5 [49.4, 52.6] | 50.0 / 98.0 | BEV 5.0 / ICE 97.9 | 63.3 / 51.0; human 64.0 / 49.4 |
|  | vehicle | 267 | 98.1 [96.6, 99.6] | 49.5 [49.0, 49.8] | 50.0 / 98.9 | BEV 0.0 / ICE 99.2 | 65.2 / 49.8; human 63.6 / 50.1 |
|  | model | 11 | 90.9 [88.9, 100.0] | 47.6 [47.1, 47.6] | 50.0 / 90.9 | BEV 0.0 / ICE 100.0 |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Nissan Leaf (2013) | BEV | 4.149 |  | BEV | 3 | 317 | 431 | 0.0 | 0.25 | 17.4 | 16.4 | 0/3 | ICE (0.26, 0.74) MISS |
| VED ICE 2500 lb | ICE | - |  | ICE | 36 | 2400 | 5534 | 0.0 | 0.66 | 70.1 | 77.0 | 35/36 | ICE (0.34, 0.66) OK |
| VED ICE 3000 lb | ICE | - |  | ICE | 53 | 3259 | 7457 | 0.0 | 0.64 | 66.7 | 71.6 | 52/53 | ICE (0.36, 0.64) OK |
| VED ICE 3500 lb | ICE | - |  | ICE | 82 | 5307 | 13279 | 0.0 | 0.65 | 68.4 | 75.1 | 81/82 | ICE (0.35, 0.65) OK |
| VED ICE 4000 lb | ICE | - |  | ICE | 29 | 1557 | 4256 | 0.0 | 0.68 | 72.3 | 79.9 | 29/29 | ICE (0.32, 0.68) OK |
| VED ICE 4500 lb | ICE | - |  | ICE | 35 | 1898 | 4611 | 0.0 | 0.67 | 71.0 | 77.3 | 32/35 | ICE (0.33, 0.67) OK |
| VED ICE 5000 lb | ICE | - |  | ICE | 5 | 154 | 577 | 0.0 | 0.69 | 72.3 | 77.9 | 5/5 | ICE (0.31, 0.69) OK |
| VED ICE 5500 lb | ICE | - |  | ICE | 4 | 109 | 331 | 0.0 | 0.66 | 69.5 | 77.1 | 4/4 | ICE (0.34, 0.66) OK |
| VED ICE 6000 lb | ICE | - |  | ICE | 3 | 36 | 198 | 0.0 | 0.68 | 70.7 | 77.8 | 3/3 | ICE (0.32, 0.68) OK |
| VED ICE 6500 lb | ICE | - |  | ICE | 1 | 5 | 15 | 0.0 | 0.76 | 93.3 | 100.0 | 1/1 | ICE (0.24, 0.76) OK |
| VED ICE weight n/a | ICE | - |  | ICE | 16 | 728 | 2570 | 0.0 | 0.68 | 71.4 | 80.8 | 16/16 | ICE (0.32, 0.68) OK |

Windows outside the bev_ice classes (scored, not in the metrics), primary run:

| model | label | veh | win | predicted share % | mean P |
|---|---|---|---|---|---|
| VED HEV 2000 lb | HEV | 1 | 164 | BEV 45.1, ICE 54.9 | 0.44, 0.56 |
| VED HEV 2500 lb | HEV | 6 | 872 | BEV 25.3, ICE 74.7 | 0.30, 0.70 |
| VED HEV 3000 lb | HEV | 52 | 9485 | BEV 25.5, ICE 74.5 | 0.30, 0.69 |
| VED HEV 3500 lb | HEV | 26 | 4423 | BEV 22.8, ICE 77.2 | 0.28, 0.72 |
| VED HEV 4000 lb | HEV | 5 | 662 | BEV 16.3, ICE 83.7 | 0.22, 0.78 |
| VED HEV 4500 lb | HEV | 2 | 400 | BEV 14.0, ICE 86.0 | 0.20, 0.80 |
| VED HEV 5000 lb | HEV | 1 | 200 | BEV 24.5, ICE 75.5 | 0.28, 0.71 |
| VED PHEV 3000 lb | PHEV | 2 | 400 | BEV 20.3, ICE 79.7 | 0.26, 0.74 |
| VED PHEV 3500 lb | PHEV | 3 | 374 | BEV 22.7, ICE 77.3 | 0.28, 0.72 |
| VED PHEV 4000 lb | PHEV | 19 | 3324 | BEV 22.9, ICE 77.1 | 0.28, 0.72 |

### ved: size (SMALL / MID / LARGE)

Windows 59563 (431 with a size label), vehicles 384, drives 25580, car models 21; label counts {'SMALL': 431, 'MID': 0, 'LARGE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `human` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; human) |
|---|---|---|---|---|---|---|---|
| V3 variant (speed-only, 3 ch) | window | 431 | 60.3 [54.8, 63.5] | - | 33.3 / 100.0 | SMALL 60.3 / MID - / LARGE - | 47.1 / 41.3; human 43.9 / 41.7 |
|  | drive | 317 | 62.5 [58.7, 65.9] | - | 33.3 / 100.0 | SMALL 62.5 / MID - / LARGE - | 53.4 / 45.1; human 50.4 / 46.5 |
|  | vehicle | 3 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 54.4 / 45.6; human 48.3 / 44.1 |
|  | model | 1 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, V3 mask (secondary) | window | 431 | 98.1 [96.8, 99.0] | - | 33.3 / 100.0 | SMALL 98.1 / MID - / LARGE - | 55.1 / 30.1; human 42.3 / 25.0 |
|  | drive | 317 | 98.1 [97.2, 100.0] | - | 33.3 / 100.0 | SMALL 98.1 / MID - / LARGE - | 52.8 / 26.1; human 47.1 / 26.1 |
|  | vehicle | 3 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 49.2 / 25.3; human 46.8 / 24.7 |
|  | model | 1 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Nissan Leaf (2013) | SMALL | 4.149 |  | BEV | 3 | 317 | 431 | 0.0 | 0.47 | 60.3 | 62.5 | 3/3 | SMALL (0.47, 0.39, 0.15) OK |

Windows outside the size classes (scored, not in the metrics), primary run:

| model | label | veh | win | predicted share % | mean P |
|---|---|---|---|---|---|
| VED HEV 2000 lb | unknown | 1 | 164 | SMALL 48.2, MID 36.6, LARGE 15.2 | 0.40, 0.37, 0.24 |
| VED HEV 2500 lb | unknown | 6 | 872 | SMALL 58.8, MID 33.6, LARGE 7.6 | 0.46, 0.36, 0.18 |
| VED HEV 3000 lb | unknown | 52 | 9485 | SMALL 53.1, MID 35.9, LARGE 11.0 | 0.43, 0.37, 0.20 |
| VED HEV 3500 lb | unknown | 26 | 4423 | SMALL 62.9, MID 32.0, LARGE 5.1 | 0.47, 0.36, 0.16 |
| VED HEV 4000 lb | unknown | 5 | 662 | SMALL 71.5, MID 23.1, LARGE 5.4 | 0.50, 0.35, 0.15 |
| VED HEV 4500 lb | unknown | 2 | 400 | SMALL 69.5, MID 26.0, LARGE 4.5 | 0.51, 0.34, 0.15 |
| VED HEV 5000 lb | unknown | 1 | 200 | SMALL 56.0, MID 36.5, LARGE 7.5 | 0.47, 0.35, 0.18 |
| VED ICE 2500 lb | unknown | 36 | 5534 | SMALL 43.3, MID 48.0, LARGE 8.7 | 0.41, 0.40, 0.19 |
| VED ICE 3000 lb | unknown | 53 | 7457 | SMALL 37.7, MID 51.2, LARGE 11.1 | 0.38, 0.40, 0.21 |
| VED ICE 3500 lb | unknown | 82 | 13279 | SMALL 39.8, MID 50.7, LARGE 9.5 | 0.39, 0.40, 0.20 |
| VED ICE 4000 lb | unknown | 29 | 4256 | SMALL 42.7, MID 50.6, LARGE 6.7 | 0.41, 0.41, 0.18 |
| VED ICE 4500 lb | unknown | 35 | 4611 | SMALL 38.4, MID 50.2, LARGE 11.3 | 0.38, 0.40, 0.21 |
| VED ICE 5000 lb | unknown | 5 | 577 | SMALL 46.8, MID 49.2, LARGE 4.0 | 0.44, 0.40, 0.17 |
| VED ICE 5500 lb | unknown | 4 | 331 | SMALL 43.5, MID 52.0, LARGE 4.5 | 0.42, 0.41, 0.17 |
| VED ICE 6000 lb | unknown | 3 | 198 | SMALL 45.5, MID 47.0, LARGE 7.6 | 0.43, 0.39, 0.18 |
| VED ICE 6500 lb | unknown | 1 | 15 | SMALL 60.0, MID 33.3, LARGE 6.7 | 0.44, 0.39, 0.17 |
| VED ICE weight n/a | unknown | 16 | 2570 | SMALL 40.4, MID 52.3, LARGE 7.4 | 0.40, 0.41, 0.19 |
| VED PHEV 3000 lb | unknown | 2 | 400 | SMALL 69.3, MID 29.5, LARGE 1.3 | 0.51, 0.38, 0.11 |
| VED PHEV 3500 lb | unknown | 3 | 374 | SMALL 60.4, MID 36.4, LARGE 3.2 | 0.48, 0.39, 0.13 |
| VED PHEV 4000 lb | unknown | 19 | 3324 | SMALL 60.3, MID 33.3, LARGE 6.5 | 0.47, 0.37, 0.16 |

## a2d2: A2D2 bus signals (Audi Q7 e-tron)

### a2d2: size (SMALL / MID / LARGE)

Windows 34 (34 with a size label), vehicles 1, drives 3, car models 1; label counts {'SMALL': 0, 'MID': 0, 'LARGE': 34}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `human` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; human) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (lead + torque absent in the data) | window | 34 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 60.0 / 54.3; human 48.9 / 48.2 |
|  | drive | 3 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 70.8 / 67.3; human 63.5 / 62.6 |
|  | vehicle | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 63.5 / 59.7; human 59.5 / 58.6 |
|  | model | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 |  |
| V1 variant (lateral + speed, 10 ch) | window | 34 | 2.9 [2.9, 2.9] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 2.9 | 62.0 / 55.0; human 55.4 / 54.1 |
|  | drive | 3 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 70.6 / 66.4; human 67.6 / 65.9 |
|  | vehicle | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 62.6 / 58.0; human 64.3 / 62.4 |
|  | model | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 |  |
| full 20-ch model, V1 mask (secondary) | window | 34 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 63.2 / 51.2; human 55.8 / 51.4 |
|  | drive | 3 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 70.5 / 63.2; human 68.4 / 65.0 |
|  | vehicle | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 60.8 / 50.8; human 64.7 / 60.0 |
|  | model | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 |  |
| V3 variant (speed-only, 3 ch) | window | 34 | 14.7 [14.7, 14.7] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 14.7 | 47.1 / 41.3; human 43.9 / 41.7 |
|  | drive | 3 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 53.4 / 45.1; human 50.4 / 46.5 |
|  | vehicle | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 54.4 / 45.6; human 48.3 / 44.1 |
|  | model | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 34 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 55.1 / 30.1; human 42.3 / 25.0 |
|  | drive | 3 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 52.8 / 26.1; human 47.1 / 26.1 |
|  | vehicle | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 | 49.2 / 25.3; human 46.8 / 24.7 |
|  | model | 1 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL - / MID - / LARGE 0.0 |  |

Per car model, primary run (full 20-ch model (lead + torque absent in the data)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Audi Q7 e-tron | LARGE | 5.045 |  | PHEV | 1 | 3 | 34 | 0.0 | 0.06 | 0.0 | 0.0 | 0/1 | SMALL (0.85, 0.09, 0.06) MISS |

## ccs_sample: commaCarSegments, five verification segments

### ccs_sample: bev_ice (BEV / ICE)

Windows 4 (4 with a bev_ice label), vehicles 4, drives 4, car models 4; label counts {'BEV': 3, 'ICE': 1}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model | window | 4 | 75.0 [25.0, 100.0] | 42.9 [20.0, 42.9] | 50.0 / 75.0 | BEV 100.0 / ICE 0.0 | 82.5 / 80.7 |
|  | drive | 4 | 75.0 [25.0, 100.0] | 42.9 [20.0, 42.9] | 50.0 / 75.0 | BEV 100.0 / ICE 0.0 | 93.7 / 93.3 |
|  | vehicle | 4 | 75.0 [25.0, 100.0] | 42.9 [20.0, 42.9] | 50.0 / 75.0 | BEV 100.0 / ICE 0.0 | 93.9 / 93.4 |
|  | model | 4 | 75.0 [50.0, 100.0] | 42.9 [33.3, 42.9] | 50.0 / 75.0 | BEV 100.0 / ICE 0.0 |  |

Per car model, primary run (full 20-ch model):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chevrolet Bolt EUV | BEV | 3.986 |  | BEV | 1 | 1 | 1 | 35.3 | 0.99 | 100.0 | 100.0 | 1/1 | BEV (0.99, 0.01) OK |
| Rivian R1T/R1S | BEV | 5.578 |  | BEV | 1 | 1 | 1 | 99.8 | 0.87 | 100.0 | 100.0 | 1/1 | BEV (0.87, 0.13) OK |
| Tesla Model 3 | BEV | 4.571 |  | BEV | 1 | 1 | 1 | 0.0 | 0.99 | 100.0 | 100.0 | 1/1 | BEV (0.99, 0.01) OK |
| Toyota RAV4 | ICE | 4.344 |  | ICE | 1 | 1 | 1 | 99.8 | 0.17 | 0.0 | 0.0 | 0/1 | BEV (0.83, 0.17) MISS |

### ccs_sample: size (SMALL / MID / LARGE)

Windows 4 (4 with a size label), vehicles 4, drives 4, car models 4; label counts {'SMALL': 2, 'MID': 1, 'LARGE': 1}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model | window | 4 | 50.0 [0.0, 100.0] | 38.9 [0.0, 100.0] | 33.3 / 50.0 | SMALL 50.0 / MID 100.0 / LARGE 0.0 | 60.0 / 54.3 |
|  | drive | 4 | 50.0 [0.0, 100.0] | 38.9 [0.0, 100.0] | 33.3 / 50.0 | SMALL 50.0 / MID 100.0 / LARGE 0.0 | 70.8 / 67.3 |
|  | vehicle | 4 | 50.0 [0.0, 100.0] | 38.9 [0.0, 100.0] | 33.3 / 50.0 | SMALL 50.0 / MID 100.0 / LARGE 0.0 | 63.5 / 59.7 |
|  | model | 4 | 50.0 [0.0, 100.0] | 38.9 [0.0, 100.0] | 33.3 / 50.0 | SMALL 50.0 / MID 100.0 / LARGE 0.0 |  |

Per car model, primary run (full 20-ch model):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chevrolet Bolt EUV | SMALL | 3.986 |  | BEV | 1 | 1 | 1 | 35.3 | 0.15 | 0.0 | 0.0 | 0/1 | MID (0.15, 0.71, 0.14) MISS |
| Rivian R1T/R1S | LARGE | 5.578 |  | BEV | 1 | 1 | 1 | 99.8 | 0.33 | 0.0 | 0.0 | 0/1 | MID (0.24, 0.43, 0.33) MISS |
| Tesla Model 3 | MID | 4.571 |  | BEV | 1 | 1 | 1 | 0.0 | 0.65 | 100.0 | 100.0 | 1/1 | MID (0.21, 0.65, 0.14) OK |
| Toyota RAV4 | SMALL | 4.344 |  | ICE | 1 | 1 | 1 | 99.8 | 0.86 | 100.0 | 100.0 | 1/1 | SMALL (0.86, 0.12, 0.02) OK |

## csc: commaSteeringControl, 36 platform archives (lateral + speed = the V1 channel list, one window per segment)

### csc: bev_ice (BEV / ICE)

Windows 51462 (51462 with a bev_ice label), vehicles 36, drives 51462, car models 25; label counts {'BEV': 11963, 'ICE': 39499}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V1 variant (lateral + speed, 10 ch) | window | 51462 | 79.4 [69.0, 88.2] | 61.7 [55.3, 67.1] | 50.0 / 76.8 | BEV 24.4 / ICE 96.1 | 70.0 / 67.9; das 69.2 / 68.0 |
|  | drive | 51462 | 79.4 [69.0, 88.2] | 61.7 [55.3, 67.1] | 50.0 / 76.8 | BEV 24.4 / ICE 96.1 | 74.6 / 73.7; das 73.9 / 73.6 |
|  | vehicle | 36 | 75.0 [61.1, 86.1] | 42.9 [37.9, 46.3] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 | 75.8 / 74.6; das 77.6 / 76.8 |
|  | model | 25 | 68.0 [60.0, 82.4] | 40.5 [37.5, 45.2] | 50.0 / 68.0 | BEV 0.0 / ICE 100.0 |  |
| full 20-ch model, V1 mask (secondary) | window | 51462 | 77.8 [66.5, 86.7] | 57.3 [53.5, 59.4] | 50.0 / 76.8 | BEV 18.3 / ICE 95.8 | 68.9 / 62.4; das 73.5 / 67.1 |
|  | drive | 51462 | 77.8 [66.5, 86.7] | 57.3 [53.5, 59.4] | 50.0 / 76.8 | BEV 18.3 / ICE 95.8 | 71.2 / 66.7; das 75.1 / 71.0 |
|  | vehicle | 36 | 75.0 [61.1, 86.1] | 42.9 [37.9, 46.3] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 | 69.3 / 61.2; das 76.6 / 71.8 |
|  | model | 25 | 68.0 [60.0, 82.4] | 40.5 [37.5, 45.2] | 50.0 / 68.0 | BEV 0.0 / ICE 100.0 |  |
| V3 variant (speed-only, 3 ch) | window | 51462 | 75.4 [72.2, 78.5] | 69.0 [61.3, 73.2] | 50.0 / 76.8 | BEV 63.9 / ICE 78.9 | 77.0 / 74.6; das 75.1 / 73.4 |
|  | drive | 51462 | 75.4 [72.2, 78.5] | 69.0 [61.3, 73.2] | 50.0 / 76.8 | BEV 63.9 / ICE 78.9 | 89.5 / 88.6; das 88.3 / 87.9 |
|  | vehicle | 36 | 97.2 [91.7, 100.0] | 96.1 [87.4, 100.0] | 50.0 / 75.0 | BEV 88.9 / ICE 100.0 | 91.0 / 90.1; das 86.5 / 85.6 |
|  | model | 25 | 96.0 [93.8, 100.0] | 95.2 [90.9, 100.0] | 50.0 / 68.0 | BEV 87.5 / ICE 100.0 |  |

Per car model, primary run (V3 variant (speed-only, 3 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chevrolet Bolt EUV | BEV | 3.986 |  | BEV | 1 | 1500 | 1500 | 90.7 | 0.66 | 73.1 | 73.1 | 1/1 | BEV (0.66, 0.34) OK |
| Chevrolet Silverado 1500 | ICE | 6.649 |  | ICE | 1 | 1500 | 1500 | 92.4 | 0.66 | 73.9 | 73.9 | 1/1 | ICE (0.34, 0.66) OK |
| Genesis GV60 | BEV | 4.734 |  | BEV | 1 | 999 | 999 | 94.9 | 0.58 | 62.8 | 62.8 | 1/1 | BEV (0.58, 0.42) OK |
| Honda Accord | ICE | 4.542 |  | ICE | 1 | 1500 | 1500 | 91.9 | 0.76 | 83.1 | 83.1 | 1/1 | ICE (0.24, 0.76) OK |
| Honda CR-V | ICE | 4.227 |  | ICE | 2 | 2500 | 2500 | 90.8 | 0.74 | 81.6 | 81.6 | 2/2 | ICE (0.26, 0.74) OK |
| Honda Civic | ICE | 4.197 |  | ICE | 3 | 4500 | 4500 | 91.3 | 0.72 | 77.9 | 77.9 | 3/3 | ICE (0.28, 0.72) OK |
| Honda Odyssey | ICE | 5.145 |  | ICE | 1 | 1500 | 1500 | 91.6 | 0.63 | 67.9 | 67.9 | 1/1 | ICE (0.37, 0.63) OK |
| Honda Pilot | ICE | 4.915 | yes | ICE | 1 | 1500 | 1500 | 92.9 | 0.78 | 86.2 | 86.2 | 1/1 | ICE (0.22, 0.78) OK |
| Hyundai Elantra | ICE | 4.311 |  | ICE | 1 | 1500 | 1500 | 90.6 | 0.69 | 74.4 | 74.4 | 1/1 | ICE (0.31, 0.69) OK |
| Hyundai Ioniq | BEV | 4.217 |  | BEV | 1 | 497 | 497 | 92.8 | 0.32 | 26.2 | 26.2 | 0/1 | ICE (0.32, 0.68) MISS |
| Hyundai Ioniq 5 | BEV | 4.826 | yes | BEV | 1 | 1500 | 1500 | 92.3 | 0.66 | 72.3 | 72.3 | 1/1 | BEV (0.66, 0.34) OK |
| Hyundai Kona | BEV | 4.025 |  | BEV | 1 | 1500 | 1500 | 92.6 | 0.55 | 58.0 | 58.0 | 1/1 | BEV (0.55, 0.45) OK |
| Hyundai Palisade | ICE | 4.965 |  | ICE | 1 | 1500 | 1500 | 93.3 | 0.67 | 73.0 | 73.0 | 1/1 | ICE (0.33, 0.67) OK |
| Hyundai Santa Fe | ICE | 4.531 |  | ICE | 2 | 3000 | 3000 | 92.9 | 0.70 | 76.9 | 76.9 | 2/2 | ICE (0.30, 0.70) OK |
| Hyundai Sonata | ICE | 4.565 |  | ICE | 2 | 3000 | 3000 | 90.7 | 0.68 | 71.9 | 71.9 | 2/2 | ICE (0.32, 0.68) OK |
| Hyundai Tucson | ICE | 4.527 |  | ICE | 1 | 999 | 999 | 93.2 | 0.75 | 83.6 | 83.6 | 1/1 | ICE (0.25, 0.75) OK |
| Kia EV6 | BEV | 4.744 |  | BEV | 1 | 1500 | 1500 | 93.1 | 0.65 | 70.4 | 70.4 | 1/1 | BEV (0.65, 0.35) OK |
| Kia Niro | BEV | 4.267 |  | BEV | 1 | 1500 | 1500 | 90.4 | 0.60 | 67.5 | 67.5 | 1/1 | BEV (0.60, 0.40) OK |
| Nissan Leaf | BEV | 4.160 |  | BEV | 2 | 2967 | 2967 | 91.0 | 0.55 | 59.6 | 59.6 | 2/2 | BEV (0.55, 0.45) OK |
| RAM 1500 | ICE | 6.732 |  | ICE | 1 | 1500 | 1500 | 94.1 | 0.67 | 73.7 | 73.7 | 1/1 | ICE (0.33, 0.67) OK |
| Toyota Camry | ICE | 4.468 | yes | ICE | 2 | 3000 | 3000 | 92.2 | 0.81 | 87.0 | 87.0 | 2/2 | ICE (0.19, 0.81) OK |
| Toyota Corolla | ICE | 4.139 |  | ICE | 2 | 3000 | 3000 | 91.4 | 0.72 | 77.3 | 77.3 | 2/2 | ICE (0.28, 0.72) OK |
| Toyota Highlander | ICE | 4.603 |  | ICE | 2 | 3000 | 3000 | 93.1 | 0.74 | 81.5 | 81.5 | 2/2 | ICE (0.26, 0.74) OK |
| Toyota RAV4 | ICE | 4.134 |  | ICE | 3 | 4500 | 4500 | 92.6 | 0.81 | 89.6 | 89.6 | 3/3 | ICE (0.19, 0.81) OK |
| Toyota Sienna | ICE | 5.212 |  | ICE | 1 | 1500 | 1500 | 93.1 | 0.60 | 63.3 | 63.3 | 1/1 | ICE (0.40, 0.60) OK |

### csc: size (SMALL / MID / LARGE)

Windows 51462 (51462 with a size label), vehicles 36, drives 51462, car models 25; label counts {'SMALL': 23964, 'MID': 18498, 'LARGE': 9000}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `das` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all; das) |
|---|---|---|---|---|---|---|---|
| V1 variant (lateral + speed, 10 ch) | window | 51462 | 48.4 [41.5, 56.1] | 44.1 [37.8, 48.6] | 33.3 / 46.6 | SMALL 68.1 / MID 29.3 / LARGE 35.1 | 62.0 / 55.0; das 61.2 / 49.8 |
|  | drive | 51462 | 48.4 [41.5, 56.1] | 44.1 [37.8, 48.6] | 33.3 / 46.6 | SMALL 68.1 / MID 29.3 / LARGE 35.1 | 70.6 / 66.4; das 65.1 / 57.4 |
|  | vehicle | 36 | 63.9 [50.0, 80.6] | 62.3 [42.9, 78.1] | 33.3 / 47.2 | SMALL 88.2 / MID 30.8 / LARGE 66.7 | 62.6 / 58.0; das 62.6 / 54.9 |
|  | model | 25 | 64.0 [50.0, 76.5] | 64.3 [45.4, 77.5] | 33.3 / 40.0 | SMALL 90.0 / MID 33.3 / LARGE 66.7 |  |
| full 20-ch model, V1 mask (secondary) | window | 51462 | 47.6 [33.8, 62.9] | 29.7 [23.9, 34.7] | 33.3 / 46.6 | SMALL 94.8 / MID 4.9 / LARGE 9.6 | 63.2 / 51.2; das 63.4 / 42.3 |
|  | drive | 51462 | 47.6 [33.8, 62.9] | 29.7 [23.9, 34.7] | 33.3 / 46.6 | SMALL 94.8 / MID 4.9 / LARGE 9.6 | 70.5 / 63.2; das 62.7 / 43.4 |
|  | vehicle | 36 | 47.2 [30.6, 63.9] | 21.4 [15.6, 26.0] | 33.3 / 47.2 | SMALL 100.0 / MID 0.0 / LARGE 0.0 | 60.8 / 50.8; das 55.3 / 36.6 |
|  | model | 25 | 40.0 [31.6, 53.3] | 19.0 [16.0, 23.2] | 33.3 / 40.0 | SMALL 100.0 / MID 0.0 / LARGE 0.0 |  |
| V3 variant (speed-only, 3 ch) | window | 51462 | 47.0 [40.4, 53.8] | 42.5 [36.1, 47.2] | 33.3 / 46.6 | SMALL 65.3 / MID 30.1 / LARGE 32.8 | 47.1 / 41.3; das 47.3 / 40.1 |
|  | drive | 51462 | 47.0 [40.4, 53.8] | 42.5 [36.1, 47.2] | 33.3 / 46.6 | SMALL 65.3 / MID 30.1 / LARGE 32.8 | 53.4 / 45.1; das 49.4 / 39.3 |
|  | vehicle | 36 | 55.6 [38.9, 72.2] | 44.9 [24.5, 57.4] | 33.3 / 47.2 | SMALL 100.0 / MID 0.0 / LARGE 50.0 | 54.4 / 45.6; das 56.1 / 47.0 |
|  | model | 25 | 52.0 [42.1, 64.7] | 43.1 [22.2, 56.4] | 33.3 / 40.0 | SMALL 100.0 / MID 0.0 / LARGE 50.0 |  |

Per car model, primary run (V1 variant (lateral + speed, 10 ch)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chevrolet Bolt EUV | SMALL | 3.986 |  | BEV | 1 | 1500 | 1500 | 90.7 | 0.54 | 70.5 | 70.5 | 1/1 | SMALL (0.54, 0.30, 0.16) OK |
| Chevrolet Silverado 1500 | LARGE | 6.649 |  | ICE | 1 | 1500 | 1500 | 92.4 | 0.38 | 37.2 | 37.2 | 1/1 | LARGE (0.28, 0.34, 0.38) OK |
| Genesis GV60 | MID | 4.734 |  | BEV | 1 | 999 | 999 | 94.9 | 0.43 | 51.2 | 51.2 | 1/1 | MID (0.38, 0.43, 0.19) OK |
| Honda Accord | MID | 4.542 |  | ICE | 1 | 1500 | 1500 | 91.9 | 0.35 | 27.4 | 27.4 | 0/1 | SMALL (0.52, 0.35, 0.13) MISS |
| Honda CR-V | SMALL | 4.227 |  | ICE | 2 | 2500 | 2500 | 90.8 | 0.48 | 63.3 | 63.3 | 1/2 | SMALL (0.48, 0.22, 0.29) OK |
| Honda Civic | SMALL | 4.197 |  | ICE | 3 | 4500 | 4500 | 91.3 | 0.58 | 78.1 | 78.1 | 3/3 | SMALL (0.58, 0.36, 0.06) OK |
| Honda Odyssey | LARGE | 5.145 |  | ICE | 1 | 1500 | 1500 | 91.6 | 0.23 | 21.7 | 21.7 | 0/1 | SMALL (0.43, 0.34, 0.23) MISS |
| Honda Pilot | LARGE | 4.915 | yes | ICE | 1 | 1500 | 1500 | 92.9 | 0.52 | 58.5 | 58.5 | 1/1 | LARGE (0.28, 0.20, 0.52) OK |
| Hyundai Elantra | SMALL | 4.311 |  | ICE | 1 | 1500 | 1500 | 90.6 | 0.56 | 72.7 | 72.7 | 1/1 | SMALL (0.56, 0.36, 0.07) OK |
| Hyundai Ioniq | SMALL | 4.217 |  | BEV | 1 | 497 | 497 | 92.8 | 0.40 | 34.8 | 34.8 | 0/1 | MID (0.40, 0.51, 0.09) MISS |
| Hyundai Ioniq 5 | MID | 4.826 | yes | BEV | 1 | 1500 | 1500 | 92.3 | 0.43 | 49.3 | 49.3 | 1/1 | MID (0.37, 0.43, 0.20) OK |
| Hyundai Kona | SMALL | 4.025 |  | BEV | 1 | 1500 | 1500 | 92.6 | 0.56 | 72.9 | 72.9 | 1/1 | SMALL (0.56, 0.35, 0.10) OK |
| Hyundai Palisade | LARGE | 4.965 |  | ICE | 1 | 1500 | 1500 | 93.3 | 0.19 | 13.6 | 13.6 | 0/1 | SMALL (0.45, 0.36, 0.19) MISS |
| Hyundai Santa Fe | MID | 4.531 |  | ICE | 2 | 3000 | 3000 | 92.9 | 0.28 | 19.4 | 19.4 | 0/2 | SMALL (0.51, 0.28, 0.21) MISS |
| Hyundai Sonata | MID | 4.565 |  | ICE | 2 | 3000 | 3000 | 90.7 | 0.32 | 20.0 | 20.0 | 0/2 | SMALL (0.53, 0.32, 0.15) MISS |
| Hyundai Tucson | MID | 4.527 |  | ICE | 1 | 999 | 999 | 93.2 | 0.45 | 50.4 | 50.4 | 1/1 | MID (0.39, 0.45, 0.16) OK |
| Kia EV6 | MID | 4.744 |  | BEV | 1 | 1500 | 1500 | 93.1 | 0.36 | 26.9 | 26.9 | 0/1 | SMALL (0.55, 0.36, 0.10) MISS |
| Kia Niro | SMALL | 4.267 |  | BEV | 1 | 1500 | 1500 | 90.4 | 0.56 | 71.7 | 71.7 | 1/1 | SMALL (0.56, 0.35, 0.09) OK |
| Nissan Leaf | SMALL | 4.160 |  | BEV | 2 | 2967 | 2967 | 91.0 | 0.53 | 61.2 | 61.2 | 2/2 | SMALL (0.53, 0.31, 0.16) OK |
| RAM 1500 | LARGE | 6.732 |  | ICE | 1 | 1500 | 1500 | 94.1 | 0.39 | 38.9 | 38.9 | 1/1 | LARGE (0.36, 0.25, 0.39) OK |
| Toyota Camry | MID | 4.468 | yes | ICE | 2 | 3000 | 3000 | 92.2 | 0.34 | 21.9 | 21.9 | 0/2 | SMALL (0.58, 0.34, 0.08) MISS |
| Toyota Corolla | SMALL | 4.139 |  | ICE | 2 | 3000 | 3000 | 91.4 | 0.50 | 60.6 | 60.6 | 2/2 | SMALL (0.50, 0.32, 0.18) OK |
| Toyota Highlander | MID | 4.603 |  | ICE | 2 | 3000 | 3000 | 93.1 | 0.36 | 33.6 | 33.6 | 1/2 | SMALL (0.42, 0.36, 0.23) MISS |
| Toyota RAV4 | SMALL | 4.134 |  | ICE | 3 | 4500 | 4500 | 92.6 | 0.53 | 69.0 | 69.0 | 3/3 | SMALL (0.53, 0.32, 0.15) OK |
| Toyota Sienna | LARGE | 5.212 |  | ICE | 1 | 1500 | 1500 | 93.1 | 0.39 | 40.9 | 40.9 | 1/1 | LARGE (0.34, 0.27, 0.39) OK |



Primary run without the borderline models (Honda Pilot, Hyundai Ioniq 5, Toyota Camry):

| level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | support |
|---|---|---|---|---|---|
| window | 45462 | 49.8 [42.1, 57.5] | 42.7 [36.6, 47.0] | SMALL 68.1 / MID 28.7 / LARGE 30.5 | 23964/13998/7500 |
| drive | 45462 | 49.8 [42.1, 57.5] | 42.7 [36.6, 47.0] | SMALL 68.1 / MID 28.7 / LARGE 30.5 | 23964/13998/7500 |
| vehicle | 32 | 65.6 [50.0, 81.2] | 60.9 [38.1, 79.3] | SMALL 88.2 / MID 30.0 / LARGE 60.0 | 17/10/5 |
| model | 22 | 63.6 [50.0, 78.6] | 61.4 [37.5, 77.5] | SMALL 90.0 / MID 28.6 / LARGE 60.0 | 10/7/5 |

## moretorque: MoreTorque/rlogs, four community-fork platforms (all 20 channels; vision lead, localizer yaw)

### moretorque: bev_ice (BEV / ICE)

Windows 7190 (7190 with a bev_ice label), vehicles 12, drives 1155, car models 4; label counts {'BEV': 874, 'ICE': 6316}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (vision lead, localizer yaw) | window | 7190 | 72.4 [48.8, 90.8] | 45.2 [37.0, 48.6] | 50.0 / 87.8 | BEV 8.0 / ICE 81.3 | 82.5 / 80.7 |
|  | drive | 1155 | 85.9 [66.1, 94.4] | 48.5 [42.4, 49.3] | 50.0 / 92.3 | BEV 4.5 / ICE 92.7 | 93.7 / 93.3 |
|  | vehicle | 12 | 91.7 [75.0, 100.0] | 47.8 [40.0, 47.8] | 50.0 / 91.7 | BEV 0.0 / ICE 100.0 | 93.9 / 93.4 |
|  | model | 4 | 75.0 [66.7, 100.0] | 42.9 [40.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 |  |
| V1 variant (lateral + speed, 10 ch) | window | 7190 | 74.6 [52.2, 99.0] | 46.5 [39.2, 53.8] | 50.0 / 87.8 | BEV 8.7 / ICE 83.8 | 70.0 / 67.9 |
|  | drive | 1155 | 84.6 [64.7, 99.3] | 48.0 [41.0, 51.9] | 50.0 / 92.3 | BEV 4.5 / ICE 91.3 | 74.6 / 73.7 |
|  | vehicle | 12 | 91.7 [75.0, 100.0] | 47.8 [40.0, 47.8] | 50.0 / 91.7 | BEV 0.0 / ICE 100.0 | 75.8 / 74.6 |
|  | model | 4 | 75.0 [66.7, 100.0] | 42.9 [40.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 |  |
| V2 variant (longitudinal-only, 11 ch) | window | 7190 | 77.7 [58.9, 89.6] | 55.1 [51.0, 59.8] | 50.0 / 87.8 | BEV 27.7 / ICE 84.6 | 76.7 / 74.1 |
|  | drive | 1155 | 87.5 [70.4, 95.4] | 55.7 [52.9, 58.5] | 50.0 / 92.3 | BEV 18.0 / ICE 93.3 | 90.0 / 89.0 |
|  | vehicle | 12 | 91.7 [75.0, 100.0] | 47.8 [40.0, 47.8] | 50.0 / 91.7 | BEV 0.0 / ICE 100.0 | 91.4 / 90.4 |
|  | model | 4 | 75.0 [66.7, 100.0] | 42.9 [40.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 |  |
| V3 variant (speed-only, 3 ch) | window | 7190 | 73.3 [59.4, 84.3] | 54.9 [49.9, 63.0] | 50.0 / 87.8 | BEV 38.9 / ICE 78.1 | 77.0 / 74.6 |
|  | drive | 1155 | 80.6 [65.1, 92.3] | 51.8 [47.2, 59.1] | 50.0 / 92.3 | BEV 21.3 / ICE 85.6 | 89.5 / 88.6 |
|  | vehicle | 12 | 91.7 [75.0, 100.0] | 47.8 [40.0, 47.8] | 50.0 / 91.7 | BEV 0.0 / ICE 100.0 | 91.0 / 90.1 |
|  | model | 4 | 75.0 [66.7, 100.0] | 42.9 [40.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 |  |
| full 20-ch model, V1 mask (secondary) | window | 7190 | 78.6 [50.1, 92.1] | 47.6 [37.2, 48.7] | 50.0 / 87.8 | BEV 7.0 / ICE 88.5 | 68.9 / 62.4 |
|  | drive | 1155 | 86.0 [65.8, 95.4] | 47.4 [40.5, 48.4] | 50.0 / 92.3 | BEV 2.2 / ICE 93.0 | 71.2 / 66.7 |
|  | vehicle | 12 | 91.7 [75.0, 100.0] | 47.8 [40.0, 47.8] | 50.0 / 91.7 | BEV 0.0 / ICE 100.0 | 69.3 / 61.2 |
|  | model | 4 | 75.0 [66.7, 100.0] | 42.9 [40.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 7190 | 82.5 [51.1, 94.8] | 49.7 [37.3, 50.6] | 50.0 / 87.8 | BEV 7.1 / ICE 93.0 | 71.0 / 60.9 |
|  | drive | 1155 | 88.6 [66.7, 97.2] | 48.4 [41.2, 49.3] | 50.0 / 92.3 | BEV 2.2 / ICE 95.8 | 67.5 / 54.4 |
|  | vehicle | 12 | 91.7 [75.0, 100.0] | 47.8 [40.0, 47.8] | 50.0 / 91.7 | BEV 0.0 / ICE 100.0 | 69.3 / 56.8 |
|  | model | 4 | 75.0 [66.7, 100.0] | 42.9 [40.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 7190 | 78.5 [49.5, 94.3] | 44.8 [32.9, 45.9] | 50.0 / 87.8 | BEV 1.5 / ICE 89.2 | 69.7 / 59.4 |
|  | drive | 1155 | 88.4 [66.8, 98.3] | 46.9 [39.3, 47.8] | 50.0 / 92.3 | BEV 0.0 / ICE 95.8 | 63.3 / 51.0 |
|  | vehicle | 12 | 91.7 [75.0, 100.0] | 47.8 [40.0, 47.8] | 50.0 / 91.7 | BEV 0.0 / ICE 100.0 | 65.2 / 49.8 |
|  | model | 4 | 75.0 [66.7, 100.0] | 42.9 [40.0, 42.9] | 50.0 / 75.0 | BEV 0.0 / ICE 100.0 |  |

Per car model, primary run (full 20-ch model (vision lead, localizer yaw)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hyundai Sonata | ICE | 4.624 |  | ICE | 1 | 297 | 2926 | 63.4 | 0.69 | 72.7 | 90.6 | 1/1 | ICE (0.31, 0.69) OK |
| Mazda 3 | ICE | 4.292 |  | ICE | 7 | 412 | 2319 | 72.8 | 0.76 | 85.6 | 91.7 | 7/7 | ICE (0.24, 0.76) OK |
| Mazda CX-50 | ICE | 4.673 |  | ICE | 3 | 357 | 1071 | 14.0 | 0.89 | 95.6 | 95.5 | 3/3 | ICE (0.11, 0.89) OK |
| Omoda E5 | BEV | 4.045 |  | BEV | 1 | 89 | 874 | 29.6 | 0.20 | 8.0 | 4.5 | 0/1 | ICE (0.20, 0.80) MISS |

### moretorque: size (SMALL / MID / LARGE)

Windows 7190 (7190 with a size label), vehicles 12, drives 1155, car models 4; label counts {'SMALL': 3193, 'MID': 3997, 'LARGE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (vision lead, localizer yaw) | window | 7190 | 43.4 [18.8, 78.9] | 41.1 [20.1, 51.0] | 33.3 / 55.6 | SMALL 81.1 / MID 13.4 / LARGE - | 60.0 / 54.3 |
|  | drive | 1155 | 42.7 [17.0, 79.6] | 37.6 [17.7, 49.5] | 33.3 / 56.6 | SMALL 85.6 / MID 9.8 / LARGE - | 70.8 / 67.3 |
|  | vehicle | 12 | 66.7 [41.7, 91.7] | 40.0 [29.4, 47.8] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - | 63.5 / 59.7 |
|  | model | 4 | 50.0 [33.3, 66.7] | 33.3 [25.0, 40.0] | 33.3 / 50.0 | SMALL 100.0 / MID 0.0 / LARGE - |  |
| V1 variant (lateral + speed, 10 ch) | window | 7190 | 44.2 [20.3, 78.0] | 41.7 [21.7, 48.8] | 33.3 / 55.6 | SMALL 82.2 / MID 13.8 / LARGE - | 62.0 / 55.0 |
|  | drive | 1155 | 40.7 [15.6, 77.5] | 35.1 [16.1, 45.9] | 33.3 / 56.6 | SMALL 83.2 / MID 8.1 / LARGE - | 70.6 / 66.4 |
|  | vehicle | 12 | 58.3 [33.3, 83.3] | 36.8 [25.0, 45.5] | 33.3 / 66.7 | SMALL 87.5 / MID 0.0 / LARGE - | 62.6 / 58.0 |
|  | model | 4 | 50.0 [33.3, 66.7] | 33.3 [25.0, 40.0] | 33.3 / 50.0 | SMALL 100.0 / MID 0.0 / LARGE - |  |
| V2 variant (longitudinal-only, 11 ch) | window | 7190 | 27.9 [15.8, 46.2] | 31.7 [18.0, 37.8] | 33.3 / 55.6 | SMALL 47.6 / MID 12.2 / LARGE - | 41.7 / 34.9 |
|  | drive | 1155 | 28.9 [11.2, 54.9] | 29.8 [13.7, 38.0] | 33.3 / 56.6 | SMALL 59.5 / MID 5.5 / LARGE - | 45.2 / 33.5 |
|  | vehicle | 12 | 41.7 [16.7, 66.7] | 31.2 [15.4, 42.1] | 33.3 / 66.7 | SMALL 62.5 / MID 0.0 / LARGE - | 43.2 / 32.8 |
|  | model | 4 | 25.0 [25.0, 50.0] | 20.0 [20.0, 33.3] | 33.3 / 50.0 | SMALL 50.0 / MID 0.0 / LARGE - |  |
| V3 variant (speed-only, 3 ch) | window | 7190 | 36.2 [16.0, 66.7] | 34.0 [16.9, 43.8] | 33.3 / 55.6 | SMALL 68.4 / MID 10.4 / LARGE - | 47.1 / 41.3 |
|  | drive | 1155 | 36.8 [12.1, 72.8] | 30.9 [12.4, 42.9] | 33.3 / 56.6 | SMALL 78.6 / MID 4.7 / LARGE - | 53.4 / 45.1 |
|  | vehicle | 12 | 66.7 [41.7, 91.7] | 40.0 [29.4, 47.8] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - | 54.4 / 45.6 |
|  | model | 4 | 50.0 [33.3, 66.7] | 33.3 [25.0, 40.0] | 33.3 / 50.0 | SMALL 100.0 / MID 0.0 / LARGE - |  |
| full 20-ch model, V1 mask (secondary) | window | 7190 | 51.5 [28.1, 84.0] | 49.4 [27.4, 57.3] | 33.3 / 55.6 | SMALL 88.4 / MID 22.0 / LARGE - | 63.2 / 51.2 |
|  | drive | 1155 | 47.2 [20.8, 80.8] | 43.1 [19.5, 59.1] | 33.3 / 56.6 | SMALL 88.2 / MID 15.7 / LARGE - | 70.5 / 63.2 |
|  | vehicle | 12 | 66.7 [41.7, 91.7] | 40.0 [29.4, 47.8] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - | 60.8 / 50.8 |
|  | model | 4 | 50.0 [33.3, 66.7] | 33.3 [25.0, 40.0] | 33.3 / 50.0 | SMALL 100.0 / MID 0.0 / LARGE - |  |
| full 20-ch model, V2 mask (secondary) | window | 7190 | 44.2 [18.6, 80.4] | 39.1 [18.8, 47.5] | 33.3 / 55.6 | SMALL 85.4 / MID 11.3 / LARGE - | 53.6 / 33.2 |
|  | drive | 1155 | 41.8 [13.3, 85.6] | 31.9 [12.9, 45.3] | 33.3 / 56.6 | SMALL 93.2 / MID 2.4 / LARGE - | 52.7 / 24.8 |
|  | vehicle | 12 | 66.7 [41.7, 91.7] | 40.0 [29.4, 47.8] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - | 48.0 / 23.2 |
|  | model | 4 | 50.0 [33.3, 66.7] | 33.3 [25.0, 40.0] | 33.3 / 50.0 | SMALL 100.0 / MID 0.0 / LARGE - |  |
| full 20-ch model, V3 mask (secondary) | window | 7190 | 43.4 [11.6, 88.1] | 33.1 [11.2, 52.7] | 33.3 / 55.6 | SMALL 93.6 / MID 3.4 / LARGE - | 55.1 / 30.1 |
|  | drive | 1155 | 42.9 [13.8, 87.7] | 32.2 [13.2, 49.2] | 33.3 / 56.6 | SMALL 95.6 / MID 2.6 / LARGE - | 52.8 / 26.1 |
|  | vehicle | 12 | 66.7 [41.7, 91.7] | 40.0 [29.4, 47.8] | 33.3 / 66.7 | SMALL 100.0 / MID 0.0 / LARGE - | 49.2 / 25.3 |
|  | model | 4 | 50.0 [33.3, 66.7] | 33.3 [25.0, 40.0] | 33.3 / 50.0 | SMALL 100.0 / MID 0.0 / LARGE - |  |

Per car model, primary run (full 20-ch model (vision lead, localizer yaw)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hyundai Sonata | MID | 4.624 |  | ICE | 1 | 297 | 2926 | 63.4 | 0.23 | 13.6 | 13.8 | 0/1 | SMALL (0.52, 0.23, 0.25) MISS |
| Mazda 3 | SMALL | 4.292 |  | ICE | 7 | 412 | 2319 | 72.8 | 0.75 | 87.7 | 91.7 | 7/7 | SMALL (0.75, 0.18, 0.07) OK |
| Mazda CX-50 | MID | 4.673 |  | ICE | 3 | 357 | 1071 | 14.0 | 0.22 | 12.6 | 6.4 | 0/3 | SMALL (0.64, 0.22, 0.14) MISS |
| Omoda E5 | SMALL | 4.045 |  | BEV | 1 | 89 | 874 | 29.6 | 0.55 | 63.6 | 57.3 | 1/1 | SMALL (0.55, 0.24, 0.20) OK |

## ccs: commaCarSegments, device-disjoint selection (all 20 channels; radar / stock-ACC / no lead by platform; steering-derived yaw)

### ccs: bev_ice (BEV / ICE)

Windows 45763 (29585 with a bev_ice label), vehicles 4061, drives 21987, car models 113; label counts {'BEV': 5420, 'ICE': 24165}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw) | window | 29585 | 81.8 [80.8, 82.8] | 74.7 [73.0, 76.4] | 50.0 / 81.7 | BEV 78.7 / ICE 82.5 | 82.5 / 80.7 |
|  | drive | 14355 | 86.6 [85.7, 87.5] | 79.4 [77.5, 81.1] | 50.0 / 83.0 | BEV 80.9 / ICE 87.8 | 93.7 / 93.3 |
|  | vehicle | 2512 | 87.8 [86.5, 89.0] | 84.7 [83.0, 86.2] | 50.0 / 75.3 | BEV 86.6 / ICE 88.2 | 93.9 / 93.4 |
|  | model | 88 | 97.7 [96.2, 100.0] | 96.0 [93.3, 100.0] | 50.0 / 83.0 | BEV 93.3 / ICE 98.6 |  |
| V1 variant (lateral + speed, 10 ch) | window | 29585 | 63.1 [61.6, 64.7] | 58.8 [56.6, 60.8] | 50.0 / 81.7 | BEV 83.9 / ICE 58.4 | 70.0 / 67.9 |
|  | drive | 14355 | 65.9 [64.2, 67.6] | 60.3 [57.7, 62.8] | 50.0 / 83.0 | BEV 83.7 / ICE 62.2 | 74.6 / 73.7 |
|  | vehicle | 2512 | 76.4 [74.9, 78.1] | 74.0 [72.3, 75.8] | 50.0 / 75.3 | BEV 92.9 / ICE 71.0 | 75.8 / 74.6 |
|  | model | 88 | 72.7 [69.9, 77.8] | 67.9 [65.4, 73.2] | 50.0 / 83.0 | BEV 100.0 / ICE 67.1 |  |
| V2 variant (longitudinal-only, 11 ch) | window | 29585 | 74.7 [73.8, 75.7] | 66.1 [64.5, 67.7] | 50.0 / 81.7 | BEV 66.5 / ICE 76.6 | 76.7 / 74.1 |
|  | drive | 14355 | 79.2 [78.1, 80.4] | 69.7 [67.8, 71.5] | 50.0 / 83.0 | BEV 68.5 / ICE 81.4 | 90.0 / 89.0 |
|  | vehicle | 2512 | 84.6 [83.3, 86.1] | 80.6 [78.9, 82.3] | 50.0 / 75.3 | BEV 78.7 / ICE 86.6 | 91.4 / 90.4 |
|  | model | 88 | 97.7 [95.1, 98.8] | 96.0 [92.2, 97.9] | 50.0 / 83.0 | BEV 93.3 / ICE 98.6 |  |
| V3 variant (speed-only, 3 ch) | window | 29585 | 81.2 [80.3, 82.1] | 72.7 [71.3, 74.1] | 50.0 / 81.7 | BEV 69.3 / ICE 83.9 | 77.0 / 74.6 |
|  | drive | 14355 | 85.1 [84.0, 86.2] | 76.2 [74.5, 77.8] | 50.0 / 83.0 | BEV 70.3 / ICE 88.1 | 89.5 / 88.6 |
|  | vehicle | 2512 | 92.6 [91.6, 93.6] | 90.1 [88.8, 91.4] | 50.0 / 75.3 | BEV 86.3 / ICE 94.7 | 91.0 / 90.1 |
|  | model | 88 | 98.9 [97.4, 98.8] | 97.9 [95.1, 97.9] | 50.0 / 83.0 | BEV 93.3 / ICE 100.0 |  |
| full 20-ch model, V1 mask (secondary) | window | 29585 | 77.0 [75.5, 78.5] | 61.9 [60.6, 63.2] | 50.0 / 81.7 | BEV 38.5 / ICE 85.6 | 68.9 / 62.4 |
|  | drive | 14355 | 78.6 [76.8, 80.4] | 61.5 [59.8, 63.1] | 50.0 / 83.0 | BEV 35.1 / ICE 87.5 | 71.2 / 66.7 |
|  | vehicle | 2512 | 82.4 [81.0, 83.9] | 71.6 [69.3, 73.7] | 50.0 / 75.3 | BEV 41.9 / ICE 95.7 | 69.3 / 61.2 |
|  | model | 88 | 86.4 [84.1, 89.9] | 62.9 [57.3, 75.5] | 50.0 / 83.0 | BEV 20.0 / ICE 100.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 29585 | 75.3 [74.0, 76.7] | 59.4 [58.3, 60.5] | 50.0 / 81.7 | BEV 34.6 / ICE 84.4 | 71.0 / 60.9 |
|  | drive | 14355 | 78.6 [76.9, 80.3] | 60.4 [58.9, 61.9] | 50.0 / 83.0 | BEV 31.8 / ICE 88.2 | 67.5 / 54.4 |
|  | vehicle | 2512 | 77.5 [75.8, 79.1] | 62.6 [60.2, 64.8] | 50.0 / 75.3 | BEV 29.1 / ICE 93.4 | 69.3 / 56.8 |
|  | model | 88 | 81.8 [80.0, 84.0] | 45.0 [44.4, 52.2] | 50.0 / 83.0 | BEV 0.0 / ICE 98.6 |  |
| full 20-ch model, V3 mask (secondary) | window | 29585 | 81.2 [79.7, 82.8] | 63.0 [61.8, 64.2] | 50.0 / 81.7 | BEV 30.3 / ICE 92.6 | 69.7 / 59.4 |
|  | drive | 14355 | 83.7 [81.8, 85.6] | 63.4 [61.8, 65.0] | 50.0 / 83.0 | BEV 27.2 / ICE 95.2 | 63.3 / 51.0 |
|  | vehicle | 2512 | 81.5 [80.0, 83.0] | 66.6 [64.2, 69.2] | 50.0 / 75.3 | BEV 29.8 / ICE 98.5 | 65.2 / 49.8 |
|  | model | 88 | 81.8 [80.0, 83.5] | 45.0 [44.4, 51.5] | 50.0 / 83.0 | BEV 0.0 / ICE 98.6 |  |
| full 20-ch model, yaw group masked (sensitivity: steering-derived yaw) | window | 29585 | 82.4 [81.5, 83.3] | 75.2 [73.5, 76.8] | 50.0 / 81.7 | BEV 77.9 / ICE 83.4 | - |
|  | drive | 14355 | 86.7 [85.8, 87.5] | 79.7 [77.7, 81.3] | 50.0 / 83.0 | BEV 82.7 / ICE 87.5 | - |
|  | vehicle | 2512 | 86.2 [84.8, 87.5] | 83.1 [81.4, 84.6] | 50.0 / 75.3 | BEV 87.4 / ICE 85.8 | - |
|  | model | 88 | 98.9 [97.5, 100.0] | 97.9 [95.7, 100.0] | 50.0 / 83.0 | BEV 93.3 / ICE 100.0 |  |
| full 20-ch model, lead group masked (sensitivity: vision-only platforms) | window | 29585 | 85.6 [84.8, 86.5] | 78.4 [76.9, 79.8] | 50.0 / 81.7 | BEV 75.5 / ICE 87.9 | - |
|  | drive | 14355 | 89.7 [88.7, 90.5] | 82.7 [81.1, 84.1] | 50.0 / 83.0 | BEV 77.5 / ICE 92.2 | - |
|  | vehicle | 2512 | 91.3 [90.2, 92.3] | 88.6 [87.1, 89.9] | 50.0 / 75.3 | BEV 85.3 / ICE 93.3 | - |
|  | model | 88 | 97.7 [96.3, 100.0] | 96.0 [93.4, 100.0] | 50.0 / 83.0 | BEV 93.3 / ICE 98.6 |  |

Per car model, primary run (full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acura ILX | ICE | 4.056 |  | ICE | 1 | 3 | 4 | 29.8 | 0.96 | 100.0 | 100.0 | 1/1 | ICE (0.04, 0.96) OK |
| Acura RDX | ICE | 4.502 |  | ICE | 21 | 208 | 476 | 35.2 | 0.83 | 88.0 | 95.2 | 21/21 | ICE (0.17, 0.83) OK |
| Audi Q3 | ICE | 4.234 |  | ICE | 11 | 192 | 380 | 17.8 | 0.84 | 88.9 | 93.2 | 11/11 | ICE (0.16, 0.84) OK |
| Buick Regal Sportback | ICE | 4.521 |  | ICE | 1 | 33 | 50 | 48.9 | 0.78 | 80.0 | 87.9 | 1/1 | ICE (0.22, 0.78) OK |
| Cadillac Escalade | ICE | 5.136 |  | ICE | 1 | 1 | 1 | 99.8 | 0.07 | 0.0 | 0.0 | 0/1 | BEV (0.93, 0.07) MISS |
| Cadillac Escalade ESV | ICE | 5.757 |  | ICE | 1 | 2 | 8 | 55.5 | 0.67 | 75.0 | 100.0 | 1/1 | ICE (0.33, 0.67) OK |
| Cadillac XT4 | ICE | 4.458 | yes | ICE | 1 | 1 | 1 | 1.2 | 0.97 | 100.0 | 100.0 | 1/1 | ICE (0.03, 0.97) OK |
| Chevrolet Bolt EUV | BEV | 3.986 |  | BEV | 160 | 162 | 380 | 43.3 | 0.73 | 71.6 | 73.5 | 118/160 | BEV (0.73, 0.27) OK |
| Chevrolet Equinox | ICE | 4.299 |  | ICE | 5 | 165 | 243 | 45.7 | 0.80 | 88.5 | 89.1 | 5/5 | ICE (0.20, 0.80) OK |
| Chevrolet Silverado 1500 | ICE | 6.649 |  | ICE | 52 | 137 | 395 | 48.3 | 0.79 | 84.3 | 94.9 | 50/52 | ICE (0.21, 0.79) OK |
| Chevrolet Traverse | ICE | 5.238 |  | ICE | 1 | 17 | 29 | 47.5 | 0.84 | 100.0 | 100.0 | 1/1 | ICE (0.16, 0.84) OK |
| Chrysler Pacifica | ICE | 5.361 |  | ICE | 59 | 346 | 709 | 28.4 | 0.83 | 88.0 | 91.6 | 56/59 | ICE (0.17, 0.83) OK |
| Dodge Durango | ICE | 4.946 |  | ICE | 1 | 1 | 11 | 0.0 | 0.84 | 90.9 | 100.0 | 1/1 | ICE (0.16, 0.84) OK |
| Ford Bronco Sport | ICE | 4.279 |  | ICE | 8 | 30 | 94 | 60.5 | 0.62 | 64.9 | 80.0 | 7/8 | ICE (0.38, 0.62) OK |
| Ford F-150 Lightning | BEV | 6.410 |  | BEV | 6 | 33 | 113 | 28.2 | 0.52 | 52.2 | 60.6 | 4/6 | BEV (0.52, 0.48) OK |
| Ford Focus | ICE | 4.216 |  | ICE | 1 | 9 | 24 | 2.9 | 0.69 | 79.2 | 77.8 | 1/1 | ICE (0.31, 0.69) OK |
| Ford Mustang Mach-E | BEV | 4.845 | yes | BEV | 12 | 142 | 305 | 34.2 | 0.80 | 83.0 | 90.8 | 11/12 | BEV (0.80, 0.20) OK |
| Ford Ranger | ICE | 5.301 |  | ICE | 3 | 36 | 82 | 51.0 | 0.67 | 72.0 | 83.3 | 3/3 | ICE (0.33, 0.67) OK |
| Genesis G70 | ICE | 4.544 |  | ICE | 4 | 41 | 128 | 15.0 | 0.82 | 90.6 | 97.6 | 4/4 | ICE (0.18, 0.82) OK |
| Genesis G80 | ICE | 4.947 |  | ICE | 7 | 148 | 304 | 45.1 | 0.73 | 76.0 | 86.5 | 7/7 | ICE (0.27, 0.73) OK |
| Genesis G90 | ICE | 5.180 |  | ICE | 9 | 125 | 249 | 48.7 | 0.73 | 74.7 | 81.6 | 9/9 | ICE (0.27, 0.73) OK |
| Genesis GV60 | BEV | 4.734 |  | BEV | 2 | 43 | 97 | 44.7 | 0.78 | 79.4 | 93.0 | 2/2 | BEV (0.78, 0.22) OK |
| Genesis GV70 | ICE | 4.733 |  | ICE | 8 | 102 | 200 | 44.1 | 0.79 | 87.0 | 89.2 | 8/8 | ICE (0.21, 0.79) OK |
| Genesis GV80 | ICE | 4.960 |  | ICE | 3 | 93 | 146 | 10.3 | 0.89 | 96.6 | 98.9 | 3/3 | ICE (0.11, 0.89) OK |
| Honda Odyssey | ICE | 5.145 |  | ICE | 96 | 176 | 381 | 28.1 | 0.81 | 85.3 | 90.9 | 89/96 | ICE (0.19, 0.81) OK |
| Honda Pilot | ICE | 4.915 | yes | ICE | 77 | 165 | 396 | 33.9 | 0.76 | 81.1 | 86.1 | 71/77 | ICE (0.24, 0.76) OK |
| Honda Ridgeline | ICE | 5.334 |  | ICE | 56 | 154 | 392 | 28.4 | 0.81 | 86.2 | 91.6 | 53/56 | ICE (0.19, 0.81) OK |
| Honda e | BEV | 3.795 |  | BEV | 1 | 1 | 2 | 99.1 | 0.34 | 50.0 | 0.0 | 0/1 | ICE (0.34, 0.66) MISS |
| Hyundai Custin | ICE | 4.861 | yes | ICE | 4 | 38 | 56 | 40.4 | 0.84 | 87.5 | 86.8 | 4/4 | ICE (0.16, 0.84) OK |
| Hyundai Elantra | ICE | 4.311 |  | ICE | 12 | 150 | 312 | 40.5 | 0.76 | 77.2 | 90.7 | 11/12 | ICE (0.24, 0.76) OK |
| Hyundai Elantra GT / i30 | ICE | 4.126 |  | ICE | 2 | 51 | 100 | 67.7 | 0.58 | 61.0 | 66.7 | 2/2 | ICE (0.42, 0.58) OK |
| Hyundai Genesis | ICE | 4.896 | yes | ICE | 13 | 165 | 279 | 27.3 | 0.77 | 81.0 | 84.8 | 13/13 | ICE (0.23, 0.77) OK |
| Hyundai Ioniq | BEV | 4.211 |  | BEV | 5 | 120 | 235 | 61.9 | 0.83 | 90.2 | 90.0 | 5/5 | BEV (0.83, 0.17) OK |
| Hyundai Ioniq 5 | BEV | 4.826 |  | BEV | 101 | 122 | 393 | 46.0 | 0.77 | 78.9 | 85.2 | 89/101 | BEV (0.77, 0.23) OK |
| Hyundai Ioniq 6 | BEV | 4.802 |  | BEV | 24 | 117 | 280 | 51.9 | 0.74 | 77.1 | 80.3 | 23/24 | BEV (0.74, 0.26) OK |
| Hyundai Kona | BEV | 4.025 |  | ICE | 20 | 256 | 538 | 46.7 | 0.75 | 80.9 | 82.4 | 16/20 | BEV (0.75, 0.25) OK |
| Hyundai Palisade | ICE | 4.965 |  | ICE | 63 | 173 | 379 | 41.4 | 0.75 | 78.4 | 86.7 | 56/63 | ICE (0.25, 0.75) OK |
| Hyundai Santa Cruz | ICE | 4.937 |  | ICE | 1 | 45 | 49 | 45.0 | 0.84 | 91.8 | 93.3 | 1/1 | ICE (0.16, 0.84) OK |
| Hyundai Santa Fe | ICE | 4.531 |  | ICE | 46 | 952 | 1792 | 39.3 | 0.73 | 76.1 | 83.4 | 43/46 | ICE (0.27, 0.73) OK |
| Hyundai Sonata | ICE | 4.624 |  | ICE | 43 | 260 | 603 | 48.6 | 0.75 | 78.3 | 84.6 | 40/43 | ICE (0.25, 0.75) OK |
| Hyundai Staria | ICE | 5.651 |  | ICE | 1 | 15 | 49 | 31.6 | 0.68 | 65.3 | 66.7 | 1/1 | ICE (0.32, 0.69) OK |
| Jeep Grand Cherokee | ICE | 4.405 | yes | ICE | 36 | 806 | 1438 | 31.5 | 0.85 | 88.7 | 92.8 | 36/36 | ICE (0.15, 0.85) OK |
| Kia Carnival | ICE | 5.302 |  | ICE | 15 | 291 | 477 | 36.6 | 0.80 | 87.2 | 92.4 | 15/15 | ICE (0.20, 0.80) OK |
| Kia EV6 | BEV | 4.744 |  | BEV | 87 | 137 | 375 | 46.0 | 0.78 | 80.5 | 84.7 | 79/87 | BEV (0.78, 0.22) OK |
| Kia Forte | ICE | 4.367 |  | ICE | 3 | 39 | 76 | 7.7 | 0.80 | 84.2 | 87.2 | 3/3 | ICE (0.20, 0.80) OK |
| Kia K5 | ICE | 4.600 |  | ICE | 3 | 67 | 99 | 37.8 | 0.77 | 84.8 | 95.5 | 3/3 | ICE (0.23, 0.77) OK |
| Kia Niro | BEV | 4.267 |  | BEV | 41 | 239 | 512 | 33.7 | 0.77 | 83.8 | 83.3 | 36/41 | BEV (0.77, 0.23) OK |
| Kia Optima | ICE | 4.529 |  | ICE | 5 | 71 | 163 | 52.7 | 0.70 | 73.0 | 69.0 | 5/5 | ICE (0.30, 0.70) OK |
| Kia Sorento | ICE | 4.639 |  | ICE | 9 | 162 | 258 | 36.8 | 0.83 | 87.2 | 89.5 | 9/9 | ICE (0.17, 0.83) OK |
| Kia Stinger | ICE | 4.508 |  | ICE | 10 | 187 | 479 | 44.9 | 0.70 | 74.1 | 81.3 | 10/10 | ICE (0.30, 0.70) OK |
| Lexus ES | ICE | 4.606 |  | ICE | 6 | 119 | 158 | 27.2 | 0.73 | 79.7 | 80.7 | 6/6 | ICE (0.27, 0.73) OK |
| Lexus IS | ICE | 4.457 | yes | ICE | 16 | 338 | 558 | 24.9 | 0.72 | 74.0 | 79.0 | 14/16 | ICE (0.28, 0.72) OK |
| Lexus LC | ICE | 4.738 |  | ICE | 2 | 53 | 67 | 14.8 | 0.88 | 94.0 | 96.2 | 2/2 | ICE (0.12, 0.88) OK |
| Lexus NX | ICE | 4.203 |  | ICE | 24 | 373 | 639 | 34.3 | 0.74 | 78.9 | 83.1 | 24/24 | ICE (0.26, 0.74) OK |
| Lexus RC | ICE | 4.327 |  | ICE | 3 | 38 | 52 | 24.5 | 0.80 | 84.6 | 86.8 | 2/3 | ICE (0.20, 0.80) OK |
| Lexus RX | ICE | 4.550 |  | ICE | 69 | 436 | 963 | 31.4 | 0.69 | 73.8 | 77.8 | 60/69 | ICE (0.30, 0.69) OK |
| Mazda CX-5 | ICE | 4.306 |  | ICE | 81 | 139 | 384 | 40.5 | 0.85 | 92.7 | 95.7 | 78/81 | ICE (0.15, 0.85) OK |
| Mazda CX-9 | ICE | 5.149 |  | ICE | 13 | 155 | 362 | 41.8 | 0.83 | 88.1 | 91.0 | 13/13 | ICE (0.17, 0.83) OK |
| Nissan Altima | ICE | 4.561 |  | ICE | 4 | 105 | 193 | 35.1 | 0.83 | 89.6 | 96.2 | 4/4 | ICE (0.17, 0.83) OK |
| Nissan Leaf | BEV | 4.160 |  | BEV | 28 | 570 | 916 | 34.2 | 0.63 | 64.0 | 68.1 | 22/28 | BEV (0.63, 0.37) OK |
| Nissan Rogue | ICE | 4.314 |  | ICE | 5 | 65 | 100 | 63.5 | 0.73 | 79.0 | 81.5 | 5/5 | ICE (0.27, 0.73) OK |
| RAM 1500 | ICE | 6.732 |  | ICE | 99 | 131 | 384 | 36.6 | 0.83 | 87.8 | 90.8 | 92/99 | ICE (0.17, 0.83) OK |
| RAM 2500/3500 HD | ICE | 6.575 |  | ICE | 5 | 19 | 74 | 19.8 | 0.82 | 83.8 | 100.0 | 5/5 | ICE (0.18, 0.82) OK |
| Rivian R1T/R1S | BEV | 5.578 |  | BEV | 96 | 96 | 386 | 44.9 | 0.88 | 92.0 | 95.8 | 92/96 | BEV (0.88, 0.12) OK |
| SEAT Ateca | ICE | 4.111 |  | ICE | 4 | 79 | 152 | 50.8 | 0.82 | 87.5 | 94.9 | 4/4 | ICE (0.18, 0.82) OK |
| Skoda Kamiq | ICE | 4.039 |  | ICE | 1 | 9 | 35 | 88.1 | 0.77 | 91.4 | 100.0 | 1/1 | ICE (0.23, 0.77) OK |
| Skoda Karoq | ICE | 4.146 |  | ICE | 3 | 8 | 8 | 16.3 | 0.90 | 100.0 | 100.0 | 3/3 | ICE (0.10, 0.90) OK |
| Skoda Kodiaq | ICE | 4.411 | yes | ICE | 12 | 166 | 296 | 32.0 | 0.81 | 88.5 | 90.4 | 12/12 | ICE (0.19, 0.81) OK |
| Skoda Octavia | ICE | 4.149 |  | ICE | 7 | 130 | 286 | 39.2 | 0.81 | 87.8 | 88.5 | 7/7 | ICE (0.19, 0.81) OK |
| Skoda Superb | ICE | 4.482 |  | ICE | 10 | 195 | 315 | 42.6 | 0.81 | 87.3 | 89.7 | 8/10 | ICE (0.19, 0.81) OK |
| Subaru Ascent | ICE | 4.721 |  | ICE | 34 | 174 | 389 | 40.6 | 0.80 | 84.3 | 89.7 | 32/34 | ICE (0.20, 0.80) OK |
| Subaru Forester | ICE | 4.185 |  | ICE | 36 | 229 | 437 | 42.0 | 0.85 | 91.8 | 95.6 | 35/36 | ICE (0.15, 0.85) OK |
| Subaru Impreza | ICE | 4.130 |  | ICE | 49 | 322 | 709 | 39.3 | 0.85 | 91.0 | 93.8 | 48/49 | ICE (0.15, 0.85) OK |
| Subaru Legacy | ICE | 4.239 |  | ICE | 2 | 23 | 51 | 34.0 | 0.86 | 88.2 | 95.7 | 2/2 | ICE (0.14, 0.86) OK |
| Subaru Outback | ICE | 4.225 |  | ICE | 46 | 208 | 491 | 40.5 | 0.84 | 88.8 | 93.8 | 46/46 | ICE (0.16, 0.84) OK |
| Tesla Model 3 | BEV | 4.571 |  | BEV | 28 | 128 | 352 | 52.9 | 0.80 | 83.8 | 83.6 | 27/28 | BEV (0.80, 0.20) OK |
| Tesla Model Y | BEV | 4.728 |  | BEV | 21 | 271 | 538 | 42.5 | 0.84 | 86.6 | 89.3 | 20/21 | BEV (0.84, 0.16) OK |
| Toyota Avalon | ICE | 4.470 | yes | ICE | 6 | 128 | 239 | 15.1 | 0.82 | 83.3 | 89.8 | 6/6 | ICE (0.18, 0.82) OK |
| Toyota Corolla | ICE | 4.139 |  | ICE | 39 | 498 | 981 | 36.8 | 0.81 | 85.6 | 91.8 | 39/39 | ICE (0.19, 0.81) OK |
| Toyota Highlander | ICE | 4.603 |  | ICE | 225 | 290 | 765 | 38.4 | 0.64 | 66.4 | 66.2 | 148/225 | ICE (0.36, 0.64) OK |
| Toyota RAV4 | ICE | 4.344 |  | ICE | 307 | 462 | 1165 | 43.2 | 0.68 | 70.8 | 78.1 | 224/307 | ICE (0.32, 0.68) OK |
| Toyota Sienna | ICE | 5.212 |  | ICE | 62 | 499 | 975 | 30.5 | 0.75 | 79.2 | 87.6 | 58/62 | ICE (0.25, 0.75) OK |
| Volkswagen Atlas | ICE | 5.108 |  | ICE | 32 | 503 | 961 | 38.0 | 0.86 | 91.8 | 94.2 | 32/32 | ICE (0.14, 0.86) OK |
| Volkswagen Jetta | ICE | 4.168 |  | ICE | 12 | 183 | 342 | 27.0 | 0.88 | 92.4 | 94.5 | 12/12 | ICE (0.12, 0.88) OK |
| Volkswagen T-Roc | ICE | 3.987 |  | ICE | 3 | 82 | 147 | 24.7 | 0.84 | 89.1 | 91.5 | 3/3 | ICE (0.16, 0.84) OK |
| Volkswagen Taos | ICE | 4.182 |  | ICE | 2 | 24 | 55 | 30.4 | 0.83 | 90.9 | 95.8 | 2/2 | ICE (0.17, 0.83) OK |
| Volkswagen Touran | ICE | 4.389 | yes | ICE | 1 | 28 | 44 | 9.2 | 0.77 | 88.6 | 92.9 | 1/1 | ICE (0.23, 0.77) OK |
| Volkswagen Transporter | ICE | 4.911 | yes | ICE | 2 | 50 | 74 | 30.4 | 0.72 | 74.3 | 80.0 | 2/2 | ICE (0.28, 0.72) OK |

Windows outside the bev_ice classes (scored, not in the metrics), primary run:

| model | label | veh | win | predicted share % | mean P |
|---|---|---|---|---|---|
| Audi A3 | unknown | 6 | 237 | BEV 7.6, ICE 92.4 | 0.14, 0.86 |
| Chevrolet Volt | PHEV | 25 | 391 | BEV 40.7, ICE 59.3 | 0.42, 0.58 |
| Chrysler Pacifica | PHEV | 125 | 766 | BEV 36.0, ICE 64.0 | 0.38, 0.62 |
| Ford Escape | unknown | 14 | 270 | BEV 36.7, ICE 63.3 | 0.39, 0.61 |
| Ford Explorer | unknown | 44 | 374 | BEV 23.3, ICE 76.7 | 0.28, 0.72 |
| Ford F-150 | unknown | 42 | 387 | BEV 15.0, ICE 85.0 | 0.21, 0.79 |
| Ford Maverick | unknown | 32 | 382 | BEV 29.3, ICE 70.7 | 0.32, 0.68 |
| Honda Accord | HEV | 120 | 390 | BEV 9.7, ICE 90.3 | 0.19, 0.81 |
| Honda CR-V | HEV | 13 | 304 | BEV 13.2, ICE 86.8 | 0.26, 0.74 |
| Honda Insight | HEV | 15 | 334 | BEV 9.9, ICE 90.1 | 0.19, 0.81 |
| Hyundai Azera | HEV | 1 | 5 | BEV 80.0, ICE 20.0 | 0.66, 0.34 |
| Hyundai Elantra | HEV | 11 | 334 | BEV 56.6, ICE 43.4 | 0.54, 0.47 |
| Hyundai Ioniq | HEV | 12 | 408 | BEV 44.9, ICE 55.1 | 0.45, 0.55 |
| Hyundai Ioniq | PHEV | 9 | 292 | BEV 51.0, ICE 49.0 | 0.50, 0.50 |
| Hyundai Ioniq Plug-in | PHEV | 2 | 100 | BEV 39.0, ICE 61.0 | 0.42, 0.58 |
| Hyundai Santa Fe | HEV | 8 | 262 | BEV 45.0, ICE 55.0 | 0.47, 0.53 |
| Hyundai Santa Fe | PHEV | 8 | 301 | BEV 52.2, ICE 47.8 | 0.52, 0.48 |
| Hyundai Sonata | HEV | 13 | 328 | BEV 65.9, ICE 34.1 | 0.61, 0.39 |
| Hyundai Tucson | unknown | 27 | 321 | BEV 24.9, ICE 75.1 | 0.31, 0.69 |
| Kia Niro | HEV | 11 | 331 | BEV 40.2, ICE 59.8 | 0.43, 0.56 |
| Kia Niro | PHEV | 5 | 158 | BEV 38.6, ICE 61.4 | 0.42, 0.58 |
| Kia Sorento | HEV | 8 | 287 | BEV 16.7, ICE 83.3 | 0.27, 0.73 |
| Kia Sportage | unknown | 9 | 277 | BEV 23.5, ICE 76.5 | 0.28, 0.72 |
| Lexus CT 200h | HEV | 3 | 65 | BEV 30.8, ICE 69.2 | 0.35, 0.65 |
| Lexus ES | HEV | 62 | 375 | BEV 37.3, ICE 62.7 | 0.40, 0.60 |
| Lexus RX | HEV | 55 | 968 | BEV 20.0, ICE 80.0 | 0.23, 0.77 |
| Toyota Alphard | unknown | 6 | 165 | BEV 20.6, ICE 79.4 | 0.25, 0.75 |
| Toyota Avalon | HEV | 25 | 774 | BEV 34.4, ICE 65.6 | 0.38, 0.62 |
| Toyota Avalon | unknown | 6 | 237 | BEV 19.8, ICE 80.2 | 0.24, 0.76 |
| Toyota C-HR | unknown | 9 | 173 | BEV 17.3, ICE 82.7 | 0.24, 0.76 |
| Toyota Camry | HEV | 167 | 781 | BEV 25.9, ICE 74.1 | 0.30, 0.70 |
| Toyota Corolla | HEV | 233 | 971 | BEV 31.9, ICE 68.1 | 0.34, 0.66 |
| Toyota Mirai | FCEV | 6 | 296 | BEV 36.5, ICE 63.5 | 0.41, 0.59 |
| Toyota Prius | HEV | 208 | 767 | BEV 41.9, ICE 58.1 | 0.44, 0.56 |
| Toyota Prius v | HEV | 3 | 50 | BEV 60.0, ICE 40.0 | 0.57, 0.43 |
| Toyota RAV4 | HEV | 104 | 773 | BEV 29.1, ICE 70.9 | 0.34, 0.66 |
| Toyota RAV4 | PHEV | 40 | 384 | BEV 59.1, ICE 40.9 | 0.56, 0.44 |
| Toyota Sienna | HEV | 32 | 879 | BEV 33.6, ICE 66.4 | 0.36, 0.64 |
| Volkswagen Arteon | unknown | 4 | 134 | BEV 18.7, ICE 81.3 | 0.28, 0.72 |
| Volkswagen Crafter | unknown | 4 | 115 | BEV 22.6, ICE 77.4 | 0.25, 0.75 |
| Volkswagen Golf | unknown | 40 | 391 | BEV 12.3, ICE 87.7 | 0.20, 0.80 |
| Volkswagen Passat | unknown | 8 | 256 | BEV 16.4, ICE 83.6 | 0.21, 0.79 |
| Volkswagen Tiguan | unknown | 31 | 385 | BEV 11.4, ICE 88.6 | 0.17, 0.83 |

### ccs: size (SMALL / MID / LARGE)

Windows 45763 (45763 with a size label), vehicles 4061, drives 21987, car models 113; label counts {'SMALL': 20036, 'MID': 15159, 'LARGE': 10568}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw) | window | 45763 | 49.6 [48.1, 51.2] | 43.2 [41.9, 44.4] | 33.3 / 43.8 | SMALL 78.1 / MID 32.6 / LARGE 20.1 | 60.0 / 54.3 |
|  | drive | 21987 | 51.3 [49.3, 53.2] | 44.0 [42.5, 45.4] | 33.3 / 45.1 | SMALL 81.6 / MID 29.7 / LARGE 21.3 | 70.8 / 67.3 |
|  | vehicle | 4061 | 54.1 [52.5, 55.7] | 46.4 [44.7, 48.2] | 33.3 / 43.4 | SMALL 91.1 / MID 27.6 / LARGE 23.5 | 63.5 / 59.7 |
|  | model | 113 | 55.8 [52.8, 58.9] | 45.6 [41.0, 50.1] | 33.3 / 44.2 | SMALL 98.0 / MID 24.2 / LARGE 20.0 |  |
| V1 variant (lateral + speed, 10 ch) | window | 45763 | 47.5 [46.0, 49.1] | 42.3 [41.0, 43.6] | 33.3 / 43.8 | SMALL 64.2 / MID 47.1 / LARGE 16.4 | 62.0 / 55.0 |
|  | drive | 21987 | 48.5 [46.7, 50.4] | 42.7 [41.1, 44.4] | 33.3 / 45.1 | SMALL 65.1 / MID 47.4 / LARGE 16.0 | 70.6 / 66.4 |
|  | vehicle | 4061 | 48.2 [46.7, 49.7] | 44.2 [42.6, 45.8] | 33.3 / 43.4 | SMALL 60.7 / MID 54.6 / LARGE 18.8 | 62.6 / 58.0 |
|  | model | 113 | 54.0 [50.5, 57.5] | 46.2 [42.0, 49.4] | 33.3 / 44.2 | SMALL 78.0 / MID 54.5 / LARGE 13.3 |  |
| V2 variant (longitudinal-only, 11 ch) | window | 45763 | 40.6 [39.4, 41.7] | 37.9 [37.0, 38.8] | 33.3 / 43.8 | SMALL 54.1 / MID 29.7 / LARGE 30.5 | 41.7 / 34.9 |
|  | drive | 21987 | 42.1 [40.6, 43.6] | 38.4 [37.2, 39.6] | 33.3 / 45.1 | SMALL 58.4 / MID 28.4 / LARGE 29.2 | 45.2 / 33.5 |
|  | vehicle | 4061 | 42.0 [40.4, 43.6] | 38.9 [37.4, 40.4] | 33.3 / 43.4 | SMALL 59.4 / MID 27.9 / LARGE 29.6 | 43.2 / 32.8 |
|  | model | 113 | 50.4 [46.2, 54.2] | 45.4 [39.3, 49.9] | 33.3 / 44.2 | SMALL 78.0 / MID 21.2 / LARGE 36.7 |  |
| V3 variant (speed-only, 3 ch) | window | 45763 | 44.7 [43.6, 45.9] | 41.7 [40.6, 42.8] | 33.3 / 43.8 | SMALL 61.7 / MID 32.2 / LARGE 30.6 | 47.1 / 41.3 |
|  | drive | 21987 | 45.7 [44.1, 47.3] | 41.4 [40.0, 42.7] | 33.3 / 45.1 | SMALL 65.6 / MID 30.1 / LARGE 28.2 | 53.4 / 45.1 |
|  | vehicle | 4061 | 53.7 [52.2, 55.3] | 50.0 [48.4, 51.5] | 33.3 / 43.4 | SMALL 80.5 / MID 24.0 / LARGE 44.5 | 54.4 / 45.6 |
|  | model | 113 | 57.5 [53.3, 60.2] | 50.2 [44.1, 53.7] | 33.3 / 44.2 | SMALL 92.0 / MID 15.2 / LARGE 46.7 |  |
| full 20-ch model, V1 mask (secondary) | window | 45763 | 48.9 [47.1, 50.9] | 36.5 [35.3, 37.8] | 33.3 / 43.8 | SMALL 91.4 / MID 19.4 / LARGE 10.6 | 63.2 / 51.2 |
|  | drive | 21987 | 50.4 [48.1, 52.8] | 36.5 [35.1, 38.1] | 33.3 / 45.1 | SMALL 93.9 / MID 17.4 / LARGE 10.7 | 70.5 / 63.2 |
|  | vehicle | 4061 | 48.2 [46.6, 49.8] | 33.2 [31.6, 34.7] | 33.3 / 43.4 | SMALL 96.8 / MID 12.9 / LARGE 8.6 | 60.8 / 50.8 |
|  | model | 113 | 45.1 [43.6, 48.6] | 22.5 [22.1, 29.2] | 33.3 / 44.2 | SMALL 100.0 / MID 3.0 / LARGE 0.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 45763 | 42.5 [40.5, 44.5] | 28.9 [27.8, 29.9] | 33.3 / 43.8 | SMALL 85.5 / MID 8.2 / LARGE 10.0 | 53.6 / 33.2 |
|  | drive | 21987 | 43.9 [41.4, 46.6] | 27.6 [26.4, 28.7] | 33.3 / 45.1 | SMALL 89.4 / MID 5.4 / LARGE 8.4 | 52.7 / 24.8 |
|  | vehicle | 4061 | 42.9 [41.3, 44.3] | 25.4 [24.3, 26.5] | 33.3 / 43.4 | SMALL 92.6 / MID 3.0 / LARGE 6.9 | 48.0 / 23.2 |
|  | model | 113 | 44.2 [42.3, 46.7] | 20.6 [19.9, 23.2] | 33.3 / 44.2 | SMALL 100.0 / MID 0.0 / LARGE 0.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 45763 | 43.4 [41.3, 45.6] | 24.8 [23.9, 25.7] | 33.3 / 43.8 | SMALL 93.5 / MID 5.6 / LARGE 2.6 | 55.1 / 30.1 |
|  | drive | 21987 | 44.4 [41.7, 47.3] | 23.5 [22.5, 24.4] | 33.3 / 45.1 | SMALL 95.1 / MID 4.0 / LARGE 1.1 | 52.8 / 26.1 |
|  | vehicle | 4061 | 43.5 [42.0, 45.1] | 21.5 [20.8, 22.3] | 33.3 / 43.4 | SMALL 98.8 / MID 2.0 / LARGE 0.1 | 49.2 / 25.3 |
|  | model | 113 | 45.1 [42.9, 46.8] | 22.6 [20.1, 23.3] | 33.3 / 44.2 | SMALL 100.0 / MID 3.0 / LARGE 0.0 |  |
| full 20-ch model, yaw group masked (sensitivity: steering-derived yaw) | window | 45763 | 35.6 [34.5, 36.8] | 34.8 [33.9, 35.8] | 33.3 / 43.8 | SMALL 27.1 / MID 22.8 / LARGE 70.1 | - |
|  | drive | 21987 | 34.8 [33.6, 36.2] | 33.9 [32.8, 35.1] | 33.3 / 45.1 | SMALL 27.4 / MID 20.5 / LARGE 71.8 | - |
|  | vehicle | 4061 | 39.8 [38.3, 41.3] | 37.5 [36.0, 39.0] | 33.3 / 43.4 | SMALL 25.7 / MID 20.4 / LARGE 88.3 | - |
|  | model | 113 | 38.1 [33.9, 41.0] | 33.3 [28.4, 37.3] | 33.3 / 44.2 | SMALL 14.0 / MID 18.2 / LARGE 100.0 |  |
| full 20-ch model, lead group masked (sensitivity: vision-only platforms) | window | 45763 | 49.3 [47.7, 51.0] | 42.1 [40.8, 43.3] | 33.3 / 43.8 | SMALL 81.3 / MID 27.2 / LARGE 20.2 | - |
|  | drive | 21987 | 50.9 [48.9, 53.0] | 42.7 [41.1, 44.1] | 33.3 / 45.1 | SMALL 84.8 / MID 24.0 / LARGE 21.6 | - |
|  | vehicle | 4061 | 53.8 [52.2, 55.3] | 45.5 [43.8, 47.1] | 33.3 / 43.4 | SMALL 93.2 / MID 23.5 / LARGE 23.8 | - |
|  | model | 113 | 55.8 [52.7, 58.5] | 44.7 [39.9, 48.9] | 33.3 / 44.2 | SMALL 100.0 / MID 21.2 / LARGE 20.0 |  |

Per car model, primary run (full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acura ILX | SMALL | 4.056 |  | ICE | 1 | 3 | 4 | 29.8 | 0.40 | 50.0 | 66.7 | 1/1 | SMALL (0.40, 0.29, 0.30) OK |
| Acura RDX | MID | 4.502 |  | ICE | 21 | 208 | 476 | 35.2 | 0.33 | 21.6 | 14.4 | 0/21 | SMALL (0.63, 0.33, 0.04) MISS |
| Audi A3 | SMALL | 3.969 |  | unknown | 6 | 127 | 237 | 34.8 | 0.73 | 88.6 | 95.3 | 5/6 | SMALL (0.73, 0.24, 0.03) OK |
| Audi Q3 | SMALL | 4.234 |  | ICE | 11 | 192 | 380 | 17.8 | 0.72 | 85.3 | 89.1 | 11/11 | SMALL (0.72, 0.25, 0.03) OK |
| Buick Regal Sportback | MID | 4.521 |  | ICE | 1 | 33 | 50 | 48.9 | 0.32 | 32.0 | 18.2 | 0/1 | SMALL (0.61, 0.32, 0.07) MISS |
| Cadillac Escalade | LARGE | 5.136 |  | ICE | 1 | 1 | 1 | 99.8 | 0.19 | 0.0 | 0.0 | 0/1 | SMALL (0.65, 0.16, 0.19) MISS |
| Cadillac Escalade ESV | LARGE | 5.757 |  | ICE | 1 | 2 | 8 | 55.5 | 0.08 | 0.0 | 0.0 | 0/1 | SMALL (0.69, 0.23, 0.08) MISS |
| Cadillac XT4 | MID | 4.458 | yes | ICE | 1 | 1 | 1 | 1.2 | 0.21 | 0.0 | 0.0 | 0/1 | SMALL (0.79, 0.21, 0.01) MISS |
| Chevrolet Bolt EUV | SMALL | 3.986 |  | BEV | 160 | 162 | 380 | 43.3 | 0.67 | 80.3 | 88.9 | 142/160 | SMALL (0.67, 0.23, 0.10) OK |
| Chevrolet Equinox | SMALL | 4.299 |  | ICE | 5 | 165 | 243 | 45.7 | 0.64 | 77.0 | 77.6 | 5/5 | SMALL (0.64, 0.31, 0.05) OK |
| Chevrolet Silverado 1500 | LARGE | 6.649 |  | ICE | 52 | 137 | 395 | 48.3 | 0.42 | 42.5 | 53.3 | 31/52 | LARGE (0.36, 0.23, 0.42) OK |
| Chevrolet Traverse | LARGE | 5.238 |  | ICE | 1 | 17 | 29 | 47.5 | 0.46 | 51.7 | 70.6 | 1/1 | LARGE (0.42, 0.12, 0.46) OK |
| Chevrolet Volt | SMALL | 4.183 |  | PHEV | 25 | 204 | 391 | 35.2 | 0.70 | 84.4 | 88.7 | 24/25 | SMALL (0.70, 0.23, 0.06) OK |
| Chrysler Pacifica | LARGE | 5.361 |  | PHEV | 184 | 686 | 1475 | 30.8 | 0.21 | 19.5 | 20.8 | 27/184 | SMALL (0.50, 0.30, 0.21) MISS |
| Dodge Durango | LARGE | 4.946 |  | ICE | 1 | 1 | 11 | 0.0 | 0.22 | 18.2 | 0.0 | 0/1 | SMALL (0.59, 0.20, 0.22) MISS |
| Ford Bronco Sport | SMALL | 4.279 |  | ICE | 8 | 30 | 94 | 60.5 | 0.42 | 44.7 | 46.7 | 3/8 | MID (0.42, 0.50, 0.08) MISS |
| Ford Escape | SMALL | 4.387 | yes | unknown | 14 | 171 | 270 | 42.2 | 0.57 | 67.8 | 67.8 | 12/14 | SMALL (0.57, 0.34, 0.09) OK |
| Ford Explorer | LARGE | 5.139 |  | unknown | 44 | 173 | 374 | 30.6 | 0.21 | 19.5 | 22.0 | 5/44 | SMALL (0.49, 0.30, 0.21) MISS |
| Ford F-150 | LARGE | 6.439 |  | unknown | 42 | 116 | 387 | 45.8 | 0.44 | 48.6 | 60.3 | 25/42 | LARGE (0.31, 0.25, 0.44) OK |
| Ford F-150 Lightning | LARGE | 6.410 |  | BEV | 6 | 33 | 113 | 28.2 | 0.55 | 65.5 | 84.8 | 5/6 | LARGE (0.26, 0.20, 0.55) OK |
| Ford Focus | SMALL | 4.216 |  | ICE | 1 | 9 | 24 | 2.9 | 0.73 | 91.7 | 88.9 | 1/1 | SMALL (0.72, 0.24, 0.03) OK |
| Ford Maverick | LARGE | 4.929 | yes | unknown | 32 | 158 | 382 | 52.3 | 0.21 | 18.3 | 20.3 | 2/32 | SMALL (0.47, 0.32, 0.21) MISS |
| Ford Mustang Mach-E | MID | 4.845 | yes | BEV | 12 | 142 | 305 | 34.2 | 0.46 | 50.8 | 59.2 | 7/12 | MID (0.40, 0.46, 0.13) OK |
| Ford Ranger | LARGE | 5.301 |  | ICE | 3 | 36 | 82 | 51.0 | 0.38 | 35.4 | 63.9 | 2/3 | SMALL (0.43, 0.19, 0.38) MISS |
| Genesis G70 | MID | 4.544 |  | ICE | 4 | 41 | 128 | 15.0 | 0.31 | 30.5 | 43.9 | 0/4 | SMALL (0.66, 0.31, 0.03) MISS |
| Genesis G80 | LARGE | 4.947 |  | ICE | 7 | 148 | 304 | 45.1 | 0.26 | 22.7 | 27.0 | 1/7 | SMALL (0.38, 0.36, 0.26) MISS |
| Genesis G90 | LARGE | 5.180 |  | ICE | 9 | 125 | 249 | 48.7 | 0.06 | 2.8 | 2.4 | 0/9 | SMALL (0.53, 0.41, 0.06) MISS |
| Genesis GV60 | MID | 4.734 |  | BEV | 2 | 43 | 97 | 44.7 | 0.52 | 59.8 | 48.8 | 1/2 | MID (0.35, 0.52, 0.13) OK |
| Genesis GV70 | MID | 4.733 |  | ICE | 8 | 102 | 200 | 44.1 | 0.36 | 29.5 | 21.6 | 1/8 | SMALL (0.57, 0.36, 0.07) MISS |
| Genesis GV80 | LARGE | 4.960 |  | ICE | 3 | 93 | 146 | 10.3 | 0.06 | 2.1 | 2.2 | 0/3 | SMALL (0.72, 0.22, 0.06) MISS |
| Honda Accord | MID | 4.542 |  | HEV | 120 | 149 | 390 | 49.0 | 0.45 | 45.1 | 45.0 | 55/120 | SMALL (0.49, 0.45, 0.07) MISS |
| Honda CR-V | SMALL | 4.277 |  | HEV | 13 | 185 | 304 | 41.8 | 0.68 | 80.3 | 89.7 | 13/13 | SMALL (0.68, 0.27, 0.05) OK |
| Honda Insight | SMALL | 4.201 |  | HEV | 15 | 181 | 334 | 51.8 | 0.72 | 85.6 | 92.3 | 14/15 | SMALL (0.71, 0.26, 0.03) OK |
| Honda Odyssey | LARGE | 5.145 |  | ICE | 96 | 176 | 381 | 28.1 | 0.08 | 3.7 | 2.8 | 2/96 | SMALL (0.56, 0.36, 0.09) MISS |
| Honda Pilot | LARGE | 4.915 | yes | ICE | 77 | 165 | 396 | 33.9 | 0.09 | 6.1 | 4.2 | 2/77 | SMALL (0.50, 0.41, 0.09) MISS |
| Honda Ridgeline | LARGE | 5.334 |  | ICE | 56 | 154 | 392 | 28.4 | 0.17 | 13.3 | 11.7 | 2/56 | SMALL (0.47, 0.36, 0.17) MISS |
| Honda e | SMALL | 3.795 |  | BEV | 1 | 1 | 2 | 99.1 | 0.75 | 100.0 | 100.0 | 1/1 | SMALL (0.75, 0.24, 0.01) OK |
| Hyundai Azera | MID | 4.633 |  | HEV | 1 | 5 | 5 | 79.9 | 0.32 | 40.0 | 40.0 | 0/1 | SMALL (0.63, 0.32, 0.06) MISS |
| Hyundai Custin | MID | 4.861 | yes | ICE | 4 | 38 | 56 | 40.4 | 0.24 | 21.4 | 21.1 | 0/4 | LARGE (0.29, 0.24, 0.47) MISS |
| Hyundai Elantra | SMALL | 4.311 |  | HEV | 23 | 374 | 646 | 41.2 | 0.65 | 79.3 | 82.4 | 23/23 | SMALL (0.65, 0.32, 0.04) OK |
| Hyundai Elantra GT / i30 | SMALL | 4.126 |  | ICE | 2 | 51 | 100 | 67.7 | 0.69 | 90.0 | 88.2 | 2/2 | SMALL (0.69, 0.28, 0.03) OK |
| Hyundai Genesis | LARGE | 4.896 | yes | ICE | 13 | 165 | 279 | 27.3 | 0.24 | 20.1 | 20.6 | 2/13 | MID (0.35, 0.41, 0.24) MISS |
| Hyundai Ioniq | SMALL | 4.217 |  | PHEV | 26 | 487 | 935 | 49.1 | 0.70 | 84.0 | 87.1 | 26/26 | SMALL (0.70, 0.27, 0.03) OK |
| Hyundai Ioniq 5 | MID | 4.826 |  | BEV | 101 | 122 | 393 | 46.0 | 0.49 | 52.4 | 46.7 | 46/101 | MID (0.47, 0.49, 0.04) OK |
| Hyundai Ioniq 6 | MID | 4.802 |  | BEV | 24 | 117 | 280 | 51.9 | 0.43 | 45.0 | 37.6 | 7/24 | SMALL (0.53, 0.43, 0.04) MISS |
| Hyundai Ioniq Plug-in | SMALL | 4.217 |  | PHEV | 2 | 35 | 100 | 43.2 | 0.70 | 93.0 | 88.6 | 2/2 | SMALL (0.70, 0.23, 0.07) OK |
| Hyundai Kona | SMALL | 4.025 |  | ICE | 20 | 256 | 538 | 46.7 | 0.63 | 70.8 | 77.3 | 18/20 | SMALL (0.63, 0.35, 0.03) OK |
| Hyundai Palisade | LARGE | 4.965 |  | ICE | 63 | 173 | 379 | 41.4 | 0.30 | 27.2 | 34.1 | 24/63 | SMALL (0.35, 0.34, 0.30) MISS |
| Hyundai Santa Cruz | LARGE | 4.937 |  | ICE | 1 | 45 | 49 | 45.0 | 0.09 | 6.1 | 6.7 | 0/1 | SMALL (0.58, 0.32, 0.09) MISS |
| Hyundai Santa Fe | MID | 4.531 |  | ICE | 62 | 1222 | 2355 | 40.4 | 0.46 | 46.5 | 48.3 | 39/62 | MID (0.44, 0.46, 0.10) OK |
| Hyundai Sonata | MID | 4.624 |  | ICE | 56 | 434 | 931 | 46.9 | 0.33 | 25.0 | 22.4 | 4/56 | SMALL (0.61, 0.33, 0.06) MISS |
| Hyundai Staria | LARGE | 5.651 |  | ICE | 1 | 15 | 49 | 31.6 | 0.04 | 2.0 | 0.0 | 0/1 | SMALL (0.64, 0.31, 0.04) MISS |
| Hyundai Tucson | MID | 4.527 |  | unknown | 27 | 146 | 321 | 41.7 | 0.31 | 24.3 | 17.8 | 3/27 | SMALL (0.64, 0.31, 0.04) MISS |
| Jeep Grand Cherokee | SMALL | 4.405 | yes | ICE | 36 | 806 | 1438 | 31.5 | 0.49 | 52.6 | 55.2 | 26/36 | SMALL (0.48, 0.39, 0.12) OK |
| Kia Carnival | LARGE | 5.302 |  | ICE | 15 | 291 | 477 | 36.6 | 0.07 | 2.7 | 2.7 | 0/15 | SMALL (0.55, 0.38, 0.07) MISS |
| Kia EV6 | MID | 4.744 |  | BEV | 87 | 137 | 375 | 46.0 | 0.63 | 71.5 | 74.5 | 70/87 | MID (0.30, 0.63, 0.07) OK |
| Kia Forte | SMALL | 4.367 |  | ICE | 3 | 39 | 76 | 7.7 | 0.72 | 84.2 | 76.9 | 3/3 | SMALL (0.72, 0.25, 0.04) OK |
| Kia K5 | MID | 4.600 |  | ICE | 3 | 67 | 99 | 37.8 | 0.26 | 9.1 | 7.5 | 0/3 | SMALL (0.71, 0.26, 0.03) MISS |
| Kia Niro | SMALL | 4.293 |  | HEV | 57 | 549 | 1001 | 42.8 | 0.68 | 81.7 | 86.0 | 53/57 | SMALL (0.68, 0.29, 0.03) OK |
| Kia Optima | MID | 4.529 |  | ICE | 5 | 71 | 163 | 52.7 | 0.30 | 18.4 | 19.7 | 0/5 | SMALL (0.67, 0.30, 0.03) MISS |
| Kia Sorento | MID | 4.639 |  | ICE | 17 | 337 | 545 | 28.8 | 0.26 | 11.4 | 9.2 | 0/17 | SMALL (0.69, 0.26, 0.05) MISS |
| Kia Sportage | SMALL | 4.421 | yes | unknown | 9 | 145 | 277 | 39.3 | 0.65 | 75.8 | 82.1 | 9/9 | SMALL (0.65, 0.31, 0.04) OK |
| Kia Stinger | MID | 4.508 |  | ICE | 10 | 187 | 479 | 44.9 | 0.56 | 62.2 | 61.0 | 10/10 | MID (0.36, 0.56, 0.09) OK |
| Lexus CT 200h | SMALL | 3.984 |  | HEV | 3 | 42 | 65 | 23.8 | 0.55 | 64.6 | 64.3 | 3/3 | SMALL (0.55, 0.34, 0.11) OK |
| Lexus ES | MID | 4.606 |  | HEV | 68 | 264 | 533 | 37.7 | 0.40 | 38.3 | 31.8 | 21/68 | SMALL (0.54, 0.41, 0.06) MISS |
| Lexus IS | SMALL | 4.457 | yes | ICE | 16 | 338 | 558 | 24.9 | 0.63 | 78.3 | 80.5 | 16/16 | SMALL (0.63, 0.32, 0.05) OK |
| Lexus LC | MID | 4.738 |  | ICE | 2 | 53 | 67 | 14.8 | 0.29 | 7.5 | 9.4 | 0/2 | SMALL (0.68, 0.29, 0.04) MISS |
| Lexus NX | SMALL | 4.203 |  | ICE | 24 | 373 | 639 | 34.3 | 0.70 | 84.0 | 88.7 | 23/24 | SMALL (0.70, 0.28, 0.02) OK |
| Lexus RC | SMALL | 4.327 |  | ICE | 3 | 38 | 52 | 24.5 | 0.66 | 86.5 | 92.1 | 3/3 | SMALL (0.66, 0.30, 0.03) OK |
| Lexus RX | MID | 4.550 |  | ICE | 124 | 883 | 1931 | 30.2 | 0.32 | 22.0 | 20.4 | 7/124 | SMALL (0.64, 0.32, 0.04) MISS |
| Mazda CX-5 | SMALL | 4.306 |  | ICE | 81 | 139 | 384 | 40.5 | 0.71 | 87.5 | 94.2 | 79/81 | SMALL (0.71, 0.25, 0.04) OK |
| Mazda CX-9 | LARGE | 5.149 |  | ICE | 13 | 155 | 362 | 41.8 | 0.36 | 36.5 | 43.9 | 4/13 | SMALL (0.48, 0.16, 0.35) MISS |
| Nissan Altima | MID | 4.561 |  | ICE | 4 | 105 | 193 | 35.1 | 0.32 | 28.0 | 17.1 | 0/4 | SMALL (0.58, 0.32, 0.10) MISS |
| Nissan Leaf | SMALL | 4.160 |  | BEV | 28 | 570 | 916 | 34.2 | 0.54 | 58.1 | 59.1 | 17/28 | SMALL (0.54, 0.43, 0.03) OK |
| Nissan Rogue | SMALL | 4.314 |  | ICE | 5 | 65 | 100 | 63.5 | 0.67 | 78.0 | 84.6 | 3/5 | SMALL (0.67, 0.28, 0.05) OK |
| RAM 1500 | LARGE | 6.732 |  | ICE | 99 | 131 | 384 | 36.6 | 0.47 | 46.1 | 58.8 | 57/99 | LARGE (0.40, 0.14, 0.47) OK |
| RAM 2500/3500 HD | LARGE | 6.575 |  | ICE | 5 | 19 | 74 | 19.8 | 0.49 | 50.0 | 63.2 | 4/5 | LARGE (0.40, 0.11, 0.49) OK |
| Rivian R1T/R1S | LARGE | 5.578 |  | BEV | 96 | 96 | 386 | 44.9 | 0.33 | 33.2 | 40.6 | 39/96 | SMALL (0.37, 0.30, 0.33) MISS |
| SEAT Ateca | SMALL | 4.111 |  | ICE | 4 | 79 | 152 | 50.8 | 0.75 | 88.2 | 96.2 | 4/4 | SMALL (0.75, 0.23, 0.02) OK |
| Skoda Kamiq | SMALL | 4.039 |  | ICE | 1 | 9 | 35 | 88.1 | 0.86 | 94.3 | 100.0 | 1/1 | SMALL (0.86, 0.11, 0.03) OK |
| Skoda Karoq | SMALL | 4.146 |  | ICE | 3 | 8 | 8 | 16.3 | 0.86 | 100.0 | 100.0 | 3/3 | SMALL (0.86, 0.11, 0.03) OK |
| Skoda Kodiaq | SMALL | 4.411 | yes | ICE | 12 | 166 | 296 | 32.0 | 0.72 | 86.8 | 91.0 | 12/12 | SMALL (0.72, 0.24, 0.04) OK |
| Skoda Octavia | SMALL | 4.149 |  | ICE | 7 | 130 | 286 | 39.2 | 0.77 | 86.7 | 91.5 | 7/7 | SMALL (0.77, 0.21, 0.02) OK |
| Skoda Superb | MID | 4.482 |  | ICE | 10 | 195 | 315 | 42.6 | 0.30 | 19.4 | 12.3 | 0/10 | SMALL (0.66, 0.30, 0.04) MISS |
| Subaru Ascent | MID | 4.721 |  | ICE | 34 | 174 | 389 | 40.6 | 0.21 | 7.2 | 5.7 | 0/34 | SMALL (0.73, 0.21, 0.06) MISS |
| Subaru Forester | SMALL | 4.185 |  | ICE | 36 | 229 | 437 | 42.0 | 0.68 | 81.7 | 84.3 | 35/36 | SMALL (0.69, 0.26, 0.06) OK |
| Subaru Impreza | SMALL | 4.130 |  | ICE | 49 | 322 | 709 | 39.3 | 0.71 | 85.6 | 87.0 | 48/49 | SMALL (0.71, 0.24, 0.05) OK |
| Subaru Legacy | SMALL | 4.239 |  | ICE | 2 | 23 | 51 | 34.0 | 0.75 | 86.3 | 91.3 | 2/2 | SMALL (0.75, 0.20, 0.06) OK |
| Subaru Outback | SMALL | 4.225 |  | ICE | 46 | 208 | 491 | 40.5 | 0.64 | 74.5 | 77.4 | 40/46 | SMALL (0.64, 0.26, 0.10) OK |
| Tesla Model 3 | MID | 4.571 |  | BEV | 28 | 128 | 352 | 52.9 | 0.53 | 59.1 | 60.9 | 17/28 | MID (0.42, 0.53, 0.05) OK |
| Tesla Model Y | MID | 4.728 |  | BEV | 21 | 271 | 538 | 42.5 | 0.59 | 66.7 | 66.8 | 17/21 | MID (0.34, 0.59, 0.06) OK |
| Toyota Alphard | MID | 4.793 |  | unknown | 6 | 82 | 165 | 34.3 | 0.33 | 18.8 | 17.1 | 0/6 | SMALL (0.61, 0.33, 0.06) MISS |
| Toyota Avalon | MID | 4.526 |  | HEV | 37 | 752 | 1250 | 32.9 | 0.29 | 17.0 | 14.8 | 1/37 | SMALL (0.67, 0.29, 0.04) MISS |
| Toyota C-HR | SMALL | 4.077 |  | unknown | 9 | 108 | 173 | 53.8 | 0.75 | 85.0 | 88.9 | 9/9 | SMALL (0.74, 0.23, 0.02) OK |
| Toyota Camry | MID | 4.468 | yes | HEV | 167 | 319 | 781 | 43.0 | 0.29 | 18.6 | 11.6 | 12/167 | SMALL (0.67, 0.29, 0.04) MISS |
| Toyota Corolla | SMALL | 4.093 |  | HEV | 272 | 875 | 1952 | 43.4 | 0.65 | 79.4 | 83.4 | 259/272 | SMALL (0.65, 0.28, 0.06) OK |
| Toyota Highlander | MID | 4.603 |  | ICE | 225 | 290 | 765 | 38.4 | 0.32 | 21.2 | 18.3 | 42/225 | SMALL (0.64, 0.32, 0.05) MISS |
| Toyota Mirai | SMALL | 4.281 |  | FCEV | 6 | 140 | 296 | 27.6 | 0.66 | 80.4 | 84.3 | 6/6 | SMALL (0.66, 0.30, 0.04) OK |
| Toyota Prius | SMALL | 4.131 |  | HEV | 208 | 271 | 767 | 49.0 | 0.72 | 87.4 | 93.7 | 198/208 | SMALL (0.72, 0.26, 0.03) OK |
| Toyota Prius v | SMALL | 4.288 |  | HEV | 3 | 17 | 50 | 56.1 | 0.63 | 78.0 | 76.5 | 3/3 | SMALL (0.63, 0.32, 0.05) OK |
| Toyota RAV4 | SMALL | 4.344 |  | ICE | 450 | 894 | 2322 | 42.9 | 0.65 | 77.1 | 82.3 | 381/450 | SMALL (0.65, 0.31, 0.04) OK |
| Toyota Sienna | LARGE | 5.212 |  | ICE | 94 | 911 | 1854 | 35.3 | 0.19 | 17.2 | 21.0 | 15/94 | SMALL (0.50, 0.31, 0.19) MISS |
| Volkswagen Arteon | MID | 4.493 |  | unknown | 4 | 43 | 134 | 52.3 | 0.22 | 5.2 | 7.0 | 0/4 | SMALL (0.76, 0.22, 0.03) MISS |
| Volkswagen Atlas | LARGE | 5.108 |  | ICE | 32 | 503 | 961 | 38.0 | 0.09 | 4.6 | 5.6 | 0/32 | SMALL (0.55, 0.37, 0.09) MISS |
| Volkswagen Crafter | LARGE | 6.239 |  | unknown | 4 | 21 | 115 | 42.4 | 0.28 | 26.1 | 42.9 | 0/4 | SMALL (0.49, 0.23, 0.28) MISS |
| Volkswagen Golf | SMALL | 4.020 |  | unknown | 40 | 178 | 391 | 46.6 | 0.73 | 83.9 | 92.7 | 38/40 | SMALL (0.73, 0.24, 0.03) OK |
| Volkswagen Jetta | SMALL | 4.168 |  | ICE | 12 | 183 | 342 | 27.0 | 0.72 | 87.7 | 89.6 | 12/12 | SMALL (0.72, 0.24, 0.04) OK |
| Volkswagen Passat | SMALL | 4.397 | yes | unknown | 8 | 125 | 256 | 46.2 | 0.68 | 81.2 | 84.0 | 8/8 | SMALL (0.68, 0.28, 0.04) OK |
| Volkswagen T-Roc | SMALL | 3.987 |  | ICE | 3 | 82 | 147 | 24.7 | 0.78 | 89.1 | 92.7 | 3/3 | SMALL (0.79, 0.16, 0.05) OK |
| Volkswagen Taos | SMALL | 4.182 |  | ICE | 2 | 24 | 55 | 30.4 | 0.75 | 92.7 | 100.0 | 2/2 | SMALL (0.75, 0.23, 0.02) OK |
| Volkswagen Tiguan | SMALL | 4.335 |  | unknown | 31 | 194 | 385 | 38.2 | 0.71 | 87.0 | 91.2 | 30/31 | SMALL (0.71, 0.24, 0.04) OK |
| Volkswagen Touran | SMALL | 4.389 | yes | ICE | 1 | 28 | 44 | 9.2 | 0.70 | 86.4 | 96.4 | 1/1 | SMALL (0.70, 0.25, 0.05) OK |
| Volkswagen Transporter | LARGE | 4.911 | yes | ICE | 2 | 50 | 74 | 30.4 | 0.07 | 4.1 | 2.0 | 0/2 | SMALL (0.57, 0.36, 0.07) MISS |



Primary run without the borderline models (Cadillac XT4, Ford Escape, Ford Maverick, Ford Mustang Mach-E, Honda Pilot, Hyundai Custin, Hyundai Genesis, Jeep Grand Cherokee, Kia Sportage, Lexus IS, Skoda Kodiaq, Toyota Avalon, Toyota Camry, Volkswagen Passat, Volkswagen Touran, Volkswagen Transporter):

| level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | recall per class % | support |
|---|---|---|---|---|---|
| window | 40428 | 50.8 [49.1, 52.4] | 44.4 [43.0, 45.8] | SMALL 80.2 / MID 33.6 / LARGE 20.8 | 17455/13536/9437 |
| drive | 19245 | 52.3 [50.2, 54.4] | 45.4 [43.6, 47.1] | SMALL 84.1 / MID 30.6 / LARGE 22.3 | 8472/6515/4258 |
| vehicle | 3678 | 57.3 [55.7, 59.1] | 49.5 [47.7, 51.5] | SMALL 91.4 / MID 30.7 / LARGE 26.0 | 1685/1085/908 |
| model | 99 | 56.6 [53.2, 60.0] | 47.3 [42.4, 52.1] | SMALL 97.7 / MID 24.1 / LARGE 23.1 | 44/29/26 |

## ccs_honda: commaCarSegments, the nine Honda platforms the first decode lost, re-decoded with the gearbox option of ccs_decode.py (opendbc 0.3.1 with CarParams.transmissionType taken from the logged gearbox message; same channels as `ccs`)

### ccs_honda: bev_ice (BEV / ICE)

Windows 2132 (2084 with a bev_ice label), vehicles 230, drives 913, car models 5; label counts {'BEV': 0, 'ICE': 2084}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw) | window | 2084 | 89.3 [86.9, 91.4] | - | 50.0 / 100.0 | BEV - / ICE 89.3 | 82.5 / 80.7 |
|  | drive | 883 | 94.1 [91.9, 96.1] | - | 50.0 / 100.0 | BEV - / ICE 94.1 | 93.7 / 93.3 |
|  | vehicle | 229 | 99.1 [97.8, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 99.1 | 93.9 / 93.4 |
|  | model | 4 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| V1 variant (lateral + speed, 10 ch) | window | 2084 | 72.7 [69.2, 76.1] | - | 50.0 / 100.0 | BEV - / ICE 72.7 | 70.0 / 67.9 |
|  | drive | 883 | 79.7 [76.3, 82.9] | - | 50.0 / 100.0 | BEV - / ICE 79.7 | 74.6 / 73.7 |
|  | vehicle | 229 | 86.0 [81.6, 90.0] | - | 50.0 / 100.0 | BEV - / ICE 86.0 | 75.8 / 74.6 |
|  | model | 4 | 100.0 [75.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| V2 variant (longitudinal-only, 11 ch) | window | 2084 | 83.6 [81.0, 86.2] | - | 50.0 / 100.0 | BEV - / ICE 83.6 | 76.7 / 74.1 |
|  | drive | 883 | 88.7 [86.1, 91.4] | - | 50.0 / 100.0 | BEV - / ICE 88.7 | 90.0 / 89.0 |
|  | vehicle | 229 | 97.8 [96.1, 99.6] | - | 50.0 / 100.0 | BEV - / ICE 97.8 | 91.4 / 90.4 |
|  | model | 4 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| V3 variant (speed-only, 3 ch) | window | 2084 | 88.8 [86.5, 90.9] | - | 50.0 / 100.0 | BEV - / ICE 88.8 | 77.0 / 74.6 |
|  | drive | 883 | 94.5 [92.4, 96.2] | - | 50.0 / 100.0 | BEV - / ICE 94.5 | 89.5 / 88.6 |
|  | vehicle | 229 | 98.3 [96.5, 99.6] | - | 50.0 / 100.0 | BEV - / ICE 98.3 | 91.0 / 90.1 |
|  | model | 4 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, V1 mask (secondary) | window | 2084 | 89.9 [87.3, 92.1] | - | 50.0 / 100.0 | BEV - / ICE 89.9 | 68.9 / 62.4 |
|  | drive | 883 | 93.0 [89.8, 95.5] | - | 50.0 / 100.0 | BEV - / ICE 93.0 | 71.2 / 66.7 |
|  | vehicle | 229 | 98.3 [96.1, 99.6] | - | 50.0 / 100.0 | BEV - / ICE 98.3 | 69.3 / 61.2 |
|  | model | 4 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 2084 | 86.3 [84.1, 88.5] | - | 50.0 / 100.0 | BEV - / ICE 86.3 | 71.0 / 60.9 |
|  | drive | 883 | 90.8 [88.8, 93.2] | - | 50.0 / 100.0 | BEV - / ICE 90.8 | 67.5 / 54.4 |
|  | vehicle | 229 | 99.6 [98.7, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 99.6 | 69.3 / 56.8 |
|  | model | 4 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 2084 | 94.0 [92.6, 95.3] | - | 50.0 / 100.0 | BEV - / ICE 94.0 | 69.7 / 59.4 |
|  | drive | 883 | 95.8 [94.5, 97.3] | - | 50.0 / 100.0 | BEV - / ICE 95.8 | 63.3 / 51.0 |
|  | vehicle | 229 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 65.2 / 49.8 |
|  | model | 4 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, yaw group masked (sensitivity: steering-derived yaw) | window | 2084 | 87.5 [85.1, 89.9] | - | 50.0 / 100.0 | BEV - / ICE 87.5 | - |
|  | drive | 883 | 91.5 [89.0, 94.0] | - | 50.0 / 100.0 | BEV - / ICE 91.5 | - |
|  | vehicle | 229 | 98.7 [97.4, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 98.7 | - |
|  | model | 4 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, lead group masked (sensitivity: vision-only platforms) | window | 2084 | 91.7 [89.8, 93.5] | - | 50.0 / 100.0 | BEV - / ICE 91.7 | - |
|  | drive | 883 | 95.8 [93.8, 97.4] | - | 50.0 / 100.0 | BEV - / ICE 95.8 | - |
|  | vehicle | 229 | 99.1 [97.8, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 99.1 | - |
|  | model | 4 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |

Per car model, primary run (full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Honda CR-V | ICE | 4.277 |  | ICE | 96 | 200 | 486 | 45.0 | 0.82 | 89.5 | 95.0 | 95/96 | ICE (0.18, 0.82) OK |
| Honda Civic | ICE | 4.197 |  | ICE | 123 | 488 | 1173 | 43.7 | 0.83 | 89.4 | 93.6 | 121/123 | ICE (0.17, 0.83) OK |
| Honda Fit | ICE | 3.737 |  | ICE | 7 | 116 | 273 | 35.0 | 0.83 | 86.8 | 94.0 | 7/7 | ICE (0.17, 0.83) OK |
| Honda HR-V | ICE | 4.010 |  | ICE | 5 | 79 | 152 | 33.3 | 0.84 | 91.4 | 94.9 | 5/5 | ICE (0.17, 0.83) OK |

Windows outside the bev_ice classes (scored, not in the metrics), primary run:

| model | label | veh | win | predicted share % | mean P |
|---|---|---|---|---|---|
| Honda Freed | unknown | 1 | 48 | BEV 16.7, ICE 83.3 | 0.20, 0.80 |

### ccs_honda: size (SMALL / MID / LARGE)

Windows 2132 (2132 with a size label), vehicles 230, drives 913, car models 5; label counts {'SMALL': 2132, 'MID': 0, 'LARGE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw) | window | 2132 | 83.8 [80.8, 86.6] | - | 33.3 / 100.0 | SMALL 83.8 / MID - / LARGE - | 60.0 / 54.3 |
|  | drive | 913 | 88.2 [84.8, 91.5] | - | 33.3 / 100.0 | SMALL 88.2 / MID - / LARGE - | 70.8 / 67.3 |
|  | vehicle | 230 | 98.3 [96.5, 99.6] | - | 33.3 / 100.0 | SMALL 98.3 / MID - / LARGE - | 63.5 / 59.7 |
|  | model | 5 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| V1 variant (lateral + speed, 10 ch) | window | 2132 | 49.6 [45.6, 53.6] | - | 33.3 / 100.0 | SMALL 49.6 / MID - / LARGE - | 62.0 / 55.0 |
|  | drive | 913 | 49.5 [43.9, 55.0] | - | 33.3 / 100.0 | SMALL 49.5 / MID - / LARGE - | 70.6 / 66.4 |
|  | vehicle | 230 | 47.8 [41.3, 54.4] | - | 33.3 / 100.0 | SMALL 47.8 / MID - / LARGE - | 62.6 / 58.0 |
|  | model | 5 | 60.0 [20.0, 60.0] | - | 33.3 / 100.0 | SMALL 60.0 / MID - / LARGE - |  |
| V2 variant (longitudinal-only, 11 ch) | window | 2132 | 51.4 [47.2, 56.4] | - | 33.3 / 100.0 | SMALL 51.4 / MID - / LARGE - | 41.7 / 34.9 |
|  | drive | 913 | 53.6 [48.6, 60.1] | - | 33.3 / 100.0 | SMALL 53.6 / MID - / LARGE - | 45.2 / 33.5 |
|  | vehicle | 230 | 73.0 [67.4, 78.7] | - | 33.3 / 100.0 | SMALL 73.0 / MID - / LARGE - | 43.2 / 32.8 |
|  | model | 5 | 60.0 [40.0, 100.0] | - | 33.3 / 100.0 | SMALL 60.0 / MID - / LARGE - |  |
| V3 variant (speed-only, 3 ch) | window | 2132 | 64.2 [60.9, 67.6] | - | 33.3 / 100.0 | SMALL 64.2 / MID - / LARGE - | 47.1 / 41.3 |
|  | drive | 913 | 72.2 [68.7, 75.9] | - | 33.3 / 100.0 | SMALL 72.2 / MID - / LARGE - | 53.4 / 45.1 |
|  | vehicle | 230 | 86.5 [82.2, 90.9] | - | 33.3 / 100.0 | SMALL 86.5 / MID - / LARGE - | 54.4 / 45.6 |
|  | model | 5 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, V1 mask (secondary) | window | 2132 | 92.5 [90.4, 94.4] | - | 33.3 / 100.0 | SMALL 92.5 / MID - / LARGE - | 63.2 / 51.2 |
|  | drive | 913 | 95.5 [93.2, 97.5] | - | 33.3 / 100.0 | SMALL 95.5 / MID - / LARGE - | 70.5 / 63.2 |
|  | vehicle | 230 | 98.7 [97.0, 100.0] | - | 33.3 / 100.0 | SMALL 98.7 / MID - / LARGE - | 60.8 / 50.8 |
|  | model | 5 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, V2 mask (secondary) | window | 2132 | 81.7 [78.3, 85.4] | - | 33.3 / 100.0 | SMALL 81.7 / MID - / LARGE - | 53.6 / 33.2 |
|  | drive | 913 | 86.2 [83.0, 89.4] | - | 33.3 / 100.0 | SMALL 86.2 / MID - / LARGE - | 52.7 / 24.8 |
|  | vehicle | 230 | 96.5 [93.9, 98.7] | - | 33.3 / 100.0 | SMALL 96.5 / MID - / LARGE - | 48.0 / 23.2 |
|  | model | 5 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, V3 mask (secondary) | window | 2132 | 89.5 [87.1, 91.8] | - | 33.3 / 100.0 | SMALL 89.5 / MID - / LARGE - | 55.1 / 30.1 |
|  | drive | 913 | 93.2 [91.0, 95.1] | - | 33.3 / 100.0 | SMALL 93.2 / MID - / LARGE - | 52.8 / 26.1 |
|  | vehicle | 230 | 97.0 [94.3, 99.1] | - | 33.3 / 100.0 | SMALL 97.0 / MID - / LARGE - | 49.2 / 25.3 |
|  | model | 5 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, yaw group masked (sensitivity: steering-derived yaw) | window | 2132 | 20.2 [17.0, 23.7] | - | 33.3 / 100.0 | SMALL 20.2 / MID - / LARGE - | - |
|  | drive | 913 | 21.7 [18.4, 25.3] | - | 33.3 / 100.0 | SMALL 21.7 / MID - / LARGE - | - |
|  | vehicle | 230 | 14.3 [10.0, 19.1] | - | 33.3 / 100.0 | SMALL 14.3 / MID - / LARGE - | - |
|  | model | 5 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL 0.0 / MID - / LARGE - |  |
| full 20-ch model, lead group masked (sensitivity: vision-only platforms) | window | 2132 | 86.2 [83.4, 88.9] | - | 33.3 / 100.0 | SMALL 86.2 / MID - / LARGE - | - |
|  | drive | 913 | 91.1 [88.0, 93.9] | - | 33.3 / 100.0 | SMALL 91.1 / MID - / LARGE - | - |
|  | vehicle | 230 | 98.7 [97.0, 100.0] | - | 33.3 / 100.0 | SMALL 98.7 / MID - / LARGE - | - |
|  | model | 5 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |

Per car model, primary run (full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Honda CR-V | SMALL | 4.277 |  | ICE | 96 | 200 | 486 | 45.0 | 0.76 | 91.6 | 93.5 | 95/96 | SMALL (0.76, 0.20, 0.04) OK |
| Honda Civic | SMALL | 4.197 |  | ICE | 123 | 488 | 1173 | 43.7 | 0.70 | 82.4 | 88.7 | 121/123 | SMALL (0.70, 0.26, 0.04) OK |
| Honda Fit | SMALL | 3.737 |  | ICE | 7 | 116 | 273 | 35.0 | 0.60 | 73.6 | 73.3 | 6/7 | SMALL (0.60, 0.36, 0.04) OK |
| Honda Freed | SMALL | 4.062 |  | unknown | 1 | 30 | 48 | 25.6 | 0.55 | 75.0 | 80.0 | 1/1 | SMALL (0.55, 0.34, 0.10) OK |
| Honda HR-V | SMALL | 4.010 |  | ICE | 5 | 79 | 152 | 33.3 | 0.73 | 90.1 | 96.2 | 5/5 | SMALL (0.73, 0.24, 0.03) OK |

## comma2k19: comma2k19, two cars on 2018 hardware (RAV4 2017, Civic 2016), decoded from the raw 2018 CAN

### comma2k19: bev_ice (BEV / ICE)

Windows 381 (381 with a bev_ice label), vehicles 2, drives 42, car models 2; label counts {'BEV': 0, 'ICE': 381}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw) | window | 381 | 70.9 [69.2, 72.4] | - | 50.0 / 100.0 | BEV - / ICE 70.9 | 82.5 / 80.7 |
|  | drive | 42 | 88.1 [85.7, 90.5] | - | 50.0 / 100.0 | BEV - / ICE 88.1 | 93.7 / 93.3 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 93.9 / 93.4 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| V1 variant (lateral + speed, 10 ch) | window | 381 | 60.4 [57.3, 63.3] | - | 50.0 / 100.0 | BEV - / ICE 60.4 | 70.0 / 67.9 |
|  | drive | 42 | 81.0 [71.4, 90.5] | - | 50.0 / 100.0 | BEV - / ICE 81.0 | 74.6 / 73.7 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 75.8 / 74.6 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| V2 variant (longitudinal-only, 11 ch) | window | 381 | 69.6 [64.3, 74.5] | - | 50.0 / 100.0 | BEV - / ICE 69.6 | 76.7 / 74.1 |
|  | drive | 42 | 83.3 [81.0, 85.7] | - | 50.0 / 100.0 | BEV - / ICE 83.3 | 90.0 / 89.0 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 91.4 / 90.4 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| V3 variant (speed-only, 3 ch) | window | 381 | 64.3 [63.8, 64.9] | - | 50.0 / 100.0 | BEV - / ICE 64.3 | 77.0 / 74.6 |
|  | drive | 42 | 83.3 [76.2, 90.5] | - | 50.0 / 100.0 | BEV - / ICE 83.3 | 89.5 / 88.6 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 91.0 / 90.1 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, V1 mask (secondary) | window | 381 | 96.6 [94.9, 98.4] | - | 50.0 / 100.0 | BEV - / ICE 96.6 | 68.9 / 62.4 |
|  | drive | 42 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 71.2 / 66.7 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 69.3 / 61.2 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, V2 mask (secondary) | window | 381 | 85.6 [83.8, 87.2] | - | 50.0 / 100.0 | BEV - / ICE 85.6 | 71.0 / 60.9 |
|  | drive | 42 | 95.2 [95.2, 95.2] | - | 50.0 / 100.0 | BEV - / ICE 95.2 | 67.5 / 54.4 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 69.3 / 56.8 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, V3 mask (secondary) | window | 381 | 94.5 [91.3, 97.8] | - | 50.0 / 100.0 | BEV - / ICE 94.5 | 69.7 / 59.4 |
|  | drive | 42 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 63.3 / 51.0 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | 65.2 / 49.8 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, yaw group masked (sensitivity: steering-derived yaw) | window | 381 | 78.0 [77.8, 78.1] | - | 50.0 / 100.0 | BEV - / ICE 78.0 | - |
|  | drive | 42 | 95.2 [95.2, 95.2] | - | 50.0 / 100.0 | BEV - / ICE 95.2 | - |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | - |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |
| full 20-ch model, lead group masked (sensitivity: vision-only platforms) | window | 381 | 92.7 [91.3, 94.1] | - | 50.0 / 100.0 | BEV - / ICE 92.7 | - |
|  | drive | 42 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | - |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 | - |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 50.0 / 100.0 | BEV - / ICE 100.0 |  |

Per car model, primary run (full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Honda Civic | ICE | 4.197 |  | ICE | 1 | 21 | 196 | 61.7 | 0.71 | 72.4 | 90.5 | 1/1 | ICE (0.29, 0.71) OK |
| Toyota RAV4 | ICE | 4.134 |  | ICE | 1 | 21 | 185 | 55.7 | 0.68 | 69.2 | 85.7 | 1/1 | ICE (0.32, 0.68) OK |

### comma2k19: size (SMALL / MID / LARGE)

Windows 381 (381 with a size label), vehicles 2, drives 42, car models 2; label counts {'SMALL': 381, 'MID': 0, 'LARGE': 0}. Internal reference = seed ensemble of the same variant on our held-out car models (accuracy / macro-F1; `all` = the matching sub-population, see internal_reference.md).

| model | level | units | accuracy % [95% CI] | macro-F1 % [95% CI] | chance uniform / majority % | recall per class % | internal reference acc / F1 (all) |
|---|---|---|---|---|---|---|---|
| full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw) | window | 381 | 51.7 [48.0, 55.7] | - | 33.3 / 100.0 | SMALL 51.7 / MID - / LARGE - | 60.0 / 54.3 |
|  | drive | 42 | 54.8 [52.4, 57.1] | - | 33.3 / 100.0 | SMALL 54.8 / MID - / LARGE - | 70.8 / 67.3 |
|  | vehicle | 2 | 50.0 [0.0, 100.0] | - | 33.3 / 100.0 | SMALL 50.0 / MID - / LARGE - | 63.5 / 59.7 |
|  | model | 2 | 50.0 [0.0, 100.0] | - | 33.3 / 100.0 | SMALL 50.0 / MID - / LARGE - |  |
| V1 variant (lateral + speed, 10 ch) | window | 381 | 44.4 [41.3, 47.6] | - | 33.3 / 100.0 | SMALL 44.4 / MID - / LARGE - | 62.0 / 55.0 |
|  | drive | 42 | 28.6 [23.8, 33.3] | - | 33.3 / 100.0 | SMALL 28.6 / MID - / LARGE - | 70.6 / 66.4 |
|  | vehicle | 2 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL 0.0 / MID - / LARGE - | 62.6 / 58.0 |
|  | model | 2 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL 0.0 / MID - / LARGE - |  |
| V2 variant (longitudinal-only, 11 ch) | window | 381 | 47.5 [40.8, 54.6] | - | 33.3 / 100.0 | SMALL 47.5 / MID - / LARGE - | 41.7 / 34.9 |
|  | drive | 42 | 57.1 [47.6, 66.7] | - | 33.3 / 100.0 | SMALL 57.1 / MID - / LARGE - | 45.2 / 33.5 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 43.2 / 32.8 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| V3 variant (speed-only, 3 ch) | window | 381 | 82.2 [81.1, 83.2] | - | 33.3 / 100.0 | SMALL 82.2 / MID - / LARGE - | 47.1 / 41.3 |
|  | drive | 42 | 97.6 [95.2, 100.0] | - | 33.3 / 100.0 | SMALL 97.6 / MID - / LARGE - | 53.4 / 45.1 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 54.4 / 45.6 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, V1 mask (secondary) | window | 381 | 97.6 [96.4, 98.9] | - | 33.3 / 100.0 | SMALL 97.6 / MID - / LARGE - | 63.2 / 51.2 |
|  | drive | 42 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 70.5 / 63.2 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 60.8 / 50.8 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, V2 mask (secondary) | window | 381 | 87.7 [87.2, 88.1] | - | 33.3 / 100.0 | SMALL 87.7 / MID - / LARGE - | 53.6 / 33.2 |
|  | drive | 42 | 97.6 [95.2, 100.0] | - | 33.3 / 100.0 | SMALL 97.6 / MID - / LARGE - | 52.7 / 24.8 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 48.0 / 23.2 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, V3 mask (secondary) | window | 381 | 96.6 [93.9, 99.5] | - | 33.3 / 100.0 | SMALL 96.6 / MID - / LARGE - | 55.1 / 30.1 |
|  | drive | 42 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 52.8 / 26.1 |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | 49.2 / 25.3 |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |
| full 20-ch model, yaw group masked (sensitivity: steering-derived yaw) | window | 381 | 3.4 [1.5, 5.4] | - | 33.3 / 100.0 | SMALL 3.4 / MID - / LARGE - | - |
|  | drive | 42 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL 0.0 / MID - / LARGE - | - |
|  | vehicle | 2 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL 0.0 / MID - / LARGE - | - |
|  | model | 2 | 0.0 [0.0, 0.0] | - | 33.3 / 100.0 | SMALL 0.0 / MID - / LARGE - |  |
| full 20-ch model, lead group masked (sensitivity: vision-only platforms) | window | 381 | 54.9 [53.0, 56.6] | - | 33.3 / 100.0 | SMALL 54.9 / MID - / LARGE - | - |
|  | drive | 42 | 59.5 [57.1, 61.9] | - | 33.3 / 100.0 | SMALL 59.5 / MID - / LARGE - | - |
|  | vehicle | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - | - |
|  | model | 2 | 100.0 [100.0, 100.0] | - | 33.3 / 100.0 | SMALL 100.0 / MID - / LARGE - |  |

Per car model, primary run (full 20-ch model (radar / stock-ACC / no lead by platform; steering-derived yaw)):

| model | label | footprint | border | powertrain | veh | drives | win | DAS % | mean P(true) | win acc % | drive acc % | veh correct | model-level decision (P) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Honda Civic | SMALL | 4.197 |  | ICE | 1 | 21 | 196 | 61.7 | 0.48 | 48.0 | 52.4 | 0/1 | MID (0.48, 0.49, 0.02) MISS |
| Toyota RAV4 | SMALL | 4.134 |  | ICE | 1 | 21 | 185 | 55.7 | 0.54 | 55.7 | 57.1 | 1/1 | SMALL (0.54, 0.44, 0.02) OK |

