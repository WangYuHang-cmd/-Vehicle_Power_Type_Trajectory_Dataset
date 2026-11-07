# Error analysis of the test predictions

Per held-out car model (headline split), per size x powertrain cell and by distance to the footprint thresholds. `dominant prediction` is the class the model assigns most often to that car model's windows.

## Task size (split_platform)

Model analysed: bi-GRU (seed 1), window macro-F1 55.1.

### Accuracy per size x powertrain cell

| size | powertrain | windows | car models | accuracy |
|---|---|---|---|---|
| LARGE | BEV | 1589 | 3 | 37.6 % |
| LARGE | HEV | 323 | 2 | 32.2 % |
| LARGE | ICE | 3463 | 9 | 30.1 % |
| MID | BEV | 2541 | 2 | 70.9 % |
| MID | HEV | 1058 | 6 | 27.6 % |
| MID | ICE | 2751 | 13 | 41.5 % |
| MID | PHEV | 18 | 2 | 94.4 % |
| MID | nan | 27 | 1 | 7.4 % |
| SMALL | BEV | 2064 | 5 | 73.6 % |
| SMALL | HEV | 2385 | 5 | 93.6 % |
| SMALL | ICE | 5868 | 9 | 71.0 % |
| SMALL | PHEV | 1517 | 3 | 89.7 % |
| SMALL | nan | 2540 | 4 | 92.4 % |

- human-only (strict): 6138 windows, accuracy 53.9 %, macro-F1 50.2
- with automation activity: 20006 windows, accuracy 66.6 %, macro-F1 56.4

### Held-out car models

| car model | true class | windows | accuracy | dominant prediction (share) | footprint m^2 |
|---|---|---|---|---|---|
| Genesis G80 | LARGE | 173 | 0.6 % | MID (77 %) | 4.95 |
| Volkswagen Atlas | LARGE | 515 | 3.1 % | SMALL (79 %) | 5.11 |
| Genesis G90 | LARGE | 346 | 4.3 % | MID (88 %) | 5.18 |
| Tesla Model S | LARGE | 374 | 6.1 % | MID (63 %) | 4.97 |
| Hyundai Santa Cruz | LARGE | 298 | 15.1 % | MID (74 %) | 4.94 |
| Toyota Sienna | LARGE | 379 | 25.6 % | SMALL (53 %) | 5.21 |
| Kia EV9 | LARGE | 863 | 30.0 % | MID (65 %) | 5.30 |
| Genesis GV80 | LARGE | 417 | 32.1 % | SMALL (35 %) | 4.96 |
| Kia Carnival | LARGE | 1190 | 40.3 % | MID (46 %) | 5.30 |
| GMC Yukon | LARGE | 359 | 76.3 % | LARGE (76 %) | 5.14 |
| Ford Ranger | LARGE | 109 | 78.0 % | LARGE (78 %) | 5.30 |
| Ford F-150 Lightning | LARGE | 352 | 89.5 % | LARGE (89 %) | 6.41 |
| Nissan Altima | MID | 144 | 7.6 % | SMALL (90 %) | 4.56 |
| Audi Q5 | MID | 101 | 8.9 % | SMALL (72 %) | 4.53 |
| Kia Sorento | MID | 404 | 8.9 % | SMALL (91 %) | 4.64 |
| GWM Haval H6 | MID | 377 | 24.1 % | SMALL (68 %) | 4.48 |
| Lexus RX | MID | 497 | 26.0 % | SMALL (61 %) | 4.55 |
| Kia Stinger | MID | 542 | 27.5 % | SMALL (71 %) | 4.51 |
| Toyota Avalon | MID | 93 | 29.0 % | SMALL (70 %) | 4.53 |
| Kia Optima | MID | 119 | 48.7 % | MID (49 %) | 4.53 |
| Hyundai Santa Fe | MID | 565 | 50.4 % | MID (50 %) | 4.57 |
| Kia K5 | MID | 115 | 56.5 % | MID (57 %) | 4.61 |
| Tesla Model Y | MID | 1150 | 63.9 % | MID (64 %) | 4.73 |
| Genesis GV70 | MID | 782 | 65.9 % | MID (66 %) | 4.73 |
| Genesis GV60 | MID | 1391 | 76.7 % | MID (77 %) | 4.73 |
| Hyundai Azera | MID | 86 | 84.9 % | MID (85 %) | 4.57 |
| Jeep Grand Cherokee | SMALL | 212 | 26.4 % | LARGE (50 %) | 4.41 |
| Chevrolet Equinox | SMALL | 165 | 26.7 % | LARGE (37 %) | 4.30 |
| Nissan Leaf | SMALL | 945 | 65.3 % | SMALL (65 %) | 4.16 |
| Hyundai Kona | SMALL | 2918 | 69.1 % | SMALL (69 %) | 4.02 |
| Honda HR-V | SMALL | 359 | 74.7 % | SMALL (75 %) | 4.17 |
| Hyundai Ioniq | SMALL | 662 | 76.4 % | SMALL (76 %) | 4.22 |
| Volkswagen Passat | SMALL | 2216 | 78.4 % | SMALL (78 %) | 4.42 |
| Volkswagen Jetta | SMALL | 431 | 80.3 % | SMALL (80 %) | 4.17 |
| Honda Clarity | SMALL | 542 | 80.8 % | SMALL (81 %) | 4.36 |
| Chevrolet Trailblazer | SMALL | 102 | 84.3 % | SMALL (84 %) | 4.10 |
| Toyota Yaris | SMALL | 99 | 85.9 % | SMALL (86 %) | 3.78 |
| Volkswagen ID.3 | SMALL | 57 | 87.7 % | SMALL (88 %) | 4.31 |
| Volkswagen ID.4 | SMALL | 227 | 89.9 % | SMALL (90 %) | 4.36 |
| Kia Ceed | SMALL | 908 | 93.9 % | SMALL (94 %) | 4.16 |
| Toyota Corolla | SMALL | 4560 | 94.8 % | SMALL (95 %) | 4.09 |

### Accuracy by distance of the footprint to the nearest threshold

| distance (m^2) | windows | car models | accuracy |
|---|---|---|---|
| (-0.001, 0.1] | 5590 | 12 | 53.0 % |
| (0.1, 0.2] | 5311 | 12 | 59.6 % |
| (0.2, 0.4] | 9667 | 14 | 75.4 % |
| (0.4, 0.8] | 5209 | 7 | 55.2 % |
| (0.8, 10.0] | 367 | 2 | 89.9 % |

## Task bev_ice (split_platform)

Model analysed: bi-GRU (seed 1), window macro-F1 80.1.

### Accuracy per size x powertrain cell

| size | powertrain | windows | car models | accuracy |
|---|---|---|---|---|
| LARGE | BEV | 1589 | 3 | 84.7 % |
| LARGE | ICE | 3463 | 9 | 76.7 % |
| MID | BEV | 2541 | 2 | 82.1 % |
| MID | ICE | 2751 | 13 | 78.0 % |
| SMALL | BEV | 2064 | 5 | 63.3 % |
| SMALL | ICE | 5868 | 9 | 92.3 % |
| nan | ICE | 19 | 2 | 94.7 % |

- human-only (strict): 4669 windows, accuracy 87.6 %, macro-F1 84.9
- with automation activity: 13626 windows, accuracy 79.9 %, macro-F1 78.4

### Held-out car models

| car model | true class | windows | accuracy | dominant prediction (share) | footprint m^2 |
|---|---|---|---|---|---|
| Nissan Leaf | BEV | 945 | 53.4 % | BEV (53 %) | 4.16 |
| Volkswagen ID.3 | BEV | 57 | 59.6 % | BEV (60 %) | 4.31 |
| Ford F-150 Lightning | BEV | 352 | 64.2 % | BEV (64 %) | 6.41 |
| Volkswagen ID.4 | BEV | 227 | 69.2 % | BEV (69 %) | 4.36 |
| Genesis GV60 | BEV | 1391 | 79.6 % | BEV (80 %) | 4.73 |
| Tesla Model S | BEV | 374 | 84.5 % | BEV (84 %) | 4.97 |
| Tesla Model Y | BEV | 1150 | 85.2 % | BEV (85 %) | 4.73 |
| Hyundai Ioniq | BEV | 219 | 85.8 % | BEV (86 %) | 4.21 |
| Kia EV9 | BEV | 863 | 93.2 % | BEV (93 %) | 5.30 |
| Genesis G80 | ICE | 173 | 39.3 % | BEV (61 %) | 4.95 |
| Hyundai Santa Cruz | ICE | 298 | 45.3 % | BEV (55 %) | 4.94 |
| Genesis G90 | ICE | 346 | 52.6 % | ICE (53 %) | 5.18 |
| Hyundai Azera | ICE | 74 | 52.7 % | ICE (53 %) | 4.57 |
| Kia Stinger | ICE | 542 | 66.8 % | ICE (67 %) | 4.51 |
| Genesis GV70 | ICE | 782 | 70.1 % | ICE (70 %) | 4.73 |
| Kia Carnival | ICE | 938 | 71.5 % | ICE (72 %) | 5.30 |
| Hyundai Santa Fe | ICE | 373 | 80.2 % | ICE (80 %) | 4.57 |
| Kia Sorento | ICE | 34 | 82.4 % | ICE (82 %) | 4.64 |
| Hyundai Kona | ICE | 2889 | 82.9 % | ICE (75 %) | 4.02 |
| Nissan Altima | ICE | 144 | 83.3 % | ICE (83 %) | 4.56 |
| Ford Ranger | ICE | 109 | 83.5 % | ICE (83 %) | 5.30 |
| Genesis GV80 | ICE | 417 | 86.1 % | ICE (86 %) | 4.96 |
| Kia Ceed | ICE | 101 | 86.1 % | ICE (86 %) | 4.15 |
| Kia Optima | ICE | 25 | 88.0 % | ICE (88 %) | 4.53 |
| Kia K5 | ICE | 115 | 91.3 % | ICE (91 %) | 4.61 |
| GWM Haval H6 | ICE | 377 | 91.5 % | ICE (92 %) | 4.48 |
| Toyota Corolla | ICE | 1031 | 91.9 % | ICE (92 %) | 4.09 |
| GMC Yukon | ICE | 373 | 94.4 % | ICE (94 %) | 5.14 |
| Chevrolet Equinox | ICE | 165 | 96.4 % | ICE (96 %) | 4.30 |
| Jeep Grand Cherokee | ICE | 212 | 96.7 % | ICE (97 %) | 4.41 |
| Honda HR-V | ICE | 282 | 96.8 % | ICE (97 %) | 4.17 |
| Audi Q5 | ICE | 101 | 97.0 % | ICE (97 %) | 4.53 |
| Chevrolet Trailblazer | ICE | 102 | 97.1 % | ICE (97 %) | 4.10 |
| Volkswagen Atlas | ICE | 515 | 97.5 % | ICE (97 %) | 5.11 |
| Lexus RX | ICE | 121 | 97.5 % | ICE (98 %) | 4.55 |
| Volkswagen Jetta | ICE | 431 | 98.1 % | ICE (98 %) | 4.17 |
| Volkswagen Passat | ICE | 1300 | 98.4 % | ICE (98 %) | 4.42 |
| Toyota Avalon | ICE | 39 | 100.0 % | ICE (100 %) | 4.47 |
| Toyota Sienna | ICE | 308 | 100.0 % | ICE (100 %) | 5.21 |
