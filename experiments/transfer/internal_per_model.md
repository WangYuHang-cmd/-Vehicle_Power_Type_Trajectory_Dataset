# Internal per-car-model decisions (split_platform test = 41 held-out car models)

Generated 9 Nov 2026 21:18 by experiments/transfer/internal_per_model.py. For every held-out car model: label, test windows / drives / vehicles, and per model the drive-level accuracy (%) and the model-level decision (mean probability over all its windows -> arg-max; P of the true class in brackets). `full+Vk` = the full model with the external mask Vk applied. These are the in-distribution counterparts of the external named-vehicle tables in RESULTS.md.

## size (SMALL / MID / LARGE); cells = drive acc % / model-level decision (P true), x = miss

| held-out model | label (footprint) | win | drives | veh | full | V1 lateral + speed | V2 longitudinal-only | V3 speed-only | full+V1 | full+V2 | full+V3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Audi Q5 | MID (4.53) | 101 | 9 | 3 | 11 / SMALL (0.24) x | 11 / SMALL (0.22) x | 11 / SMALL (0.30) x | 78 / MID (0.38) | 0 / SMALL (0.10) x | 0 / SMALL (0.15) x | 11 / SMALL (0.21) x |
| Chevrolet Equinox | SMALL (4.30) | 165 | 12 | 3 | 50 / SMALL (0.37) | 25 / MID (0.25) x | 75 / SMALL (0.46) | 83 / SMALL (0.50) | 92 / SMALL (0.58) | 100 / SMALL (0.78) | 100 / SMALL (0.61) |
| Chevrolet Trailblazer | SMALL (4.10) | 102 | 8 | 3 | 88 / SMALL (0.70) | 75 / SMALL (0.69) | 75 / SMALL (0.41) | 100 / SMALL (0.49) | 100 / SMALL (0.80) | 100 / SMALL (0.82) | 100 / SMALL (0.64) |
| Ford F-150 Lightning | LARGE (6.41) | 352 | 22 | 3 | 95 / LARGE (0.73) | 100 / LARGE (0.67) | 18 / LARGE (0.35) | 18 / SMALL (0.37) x | 73 / SMALL (0.42) x | 5 / SMALL (0.21) x | 0 / SMALL (0.24) x |
| Ford Ranger | LARGE (5.30) | 109 | 10 | 5 | 90 / LARGE (0.74) | 90 / LARGE (0.58) | 70 / LARGE (0.43) | 40 / LARGE (0.45) | 30 / SMALL (0.42) x | 0 / SMALL (0.17) x | 0 / SMALL (0.20) x |
| GMC Yukon | LARGE (5.17) | 359 | 12 | 4 | 100 / LARGE (0.64) | 75 / LARGE (0.52) | 33 / SMALL (0.33) x | 100 / LARGE (0.61) | 50 / SMALL (0.34) x | 0 / SMALL (0.23) x | 0 / SMALL (0.24) x |
| GWM Haval H6 | MID (4.48) | 377 | 17 | 3 | 6 / SMALL (0.31) x | 6 / SMALL (0.31) x | 6 / SMALL (0.31) x | 0 / SMALL (0.30) x | 0 / SMALL (0.24) x | 0 / SMALL (0.12) x | 12 / SMALL (0.21) x |
| Genesis G80 | LARGE (4.95) | 173 | 5 | 3 | 0 / MID (0.13) x | 0 / MID (0.17) x | 20 / SMALL (0.33) x | 20 / MID (0.30) x | 0 / MID (0.11) x | 0 / SMALL (0.21) x | 20 / SMALL (0.24) x |
| Genesis G90 | LARGE (5.21) | 346 | 9 | 3 | 0 / MID (0.13) x | 0 / MID (0.10) x | 0 / SMALL (0.28) x | 0 / MID (0.25) x | 0 / MID (0.08) x | 0 / SMALL (0.17) x | 0 / SMALL (0.24) x |
| Genesis GV60 | MID (4.73) | 1391 | 69 | 3 | 97 / MID (0.61) | 84 / SMALL (0.44) x | 3 / SMALL (0.31) x | 36 / SMALL (0.37) x | 78 / MID (0.49) | 6 / SMALL (0.26) x | 3 / SMALL (0.27) x |
| Genesis GV70 | MID (4.73) | 782 | 24 | 9 | 67 / MID (0.45) | 75 / MID (0.53) | 8 / SMALL (0.26) x | 50 / LARGE (0.28) x | 75 / MID (0.41) | 0 / SMALL (0.18) x | 0 / SMALL (0.23) x |
| Genesis GV80 | LARGE (4.96) | 417 | 37 | 8 | 62 / LARGE (0.51) | 62 / LARGE (0.54) | 62 / LARGE (0.47) | 76 / LARGE (0.57) | 57 / LARGE (0.47) | 0 / SMALL (0.23) x | 0 / SMALL (0.18) x |
| Honda Clarity | SMALL (4.36) | 542 | 29 | 6 | 76 / SMALL (0.48) | 86 / SMALL (0.55) | 31 / MID (0.34) x | 62 / SMALL (0.42) | 100 / SMALL (0.68) | 97 / SMALL (0.60) | 100 / SMALL (0.60) |
| Honda HR-V | SMALL (4.13) | 359 | 24 | 9 | 46 / SMALL (0.50) | 58 / SMALL (0.48) | 75 / SMALL (0.45) | 83 / SMALL (0.50) | 88 / SMALL (0.74) | 96 / SMALL (0.75) | 100 / SMALL (0.61) |
| Hyundai Azera | MID (4.58) | 86 | 5 | 3 | 80 / MID (0.50) | 100 / MID (0.66) | 0 / LARGE (0.25) x | 0 / SMALL (0.28) x | 80 / MID (0.48) | 0 / SMALL (0.10) x | 0 / SMALL (0.16) x |
| Hyundai Ioniq | SMALL (4.22) | 662 | 35 | 13 | 69 / SMALL (0.59) | 63 / SMALL (0.58) | 51 / SMALL (0.40) | 51 / SMALL (0.43) | 80 / SMALL (0.60) | 100 / SMALL (0.73) | 97 / SMALL (0.64) |
| Hyundai Kona | SMALL (4.03) | 2918 | 125 | 25 | 86 / SMALL (0.60) | 91 / SMALL (0.69) | 82 / SMALL (0.43) | 67 / SMALL (0.42) | 94 / SMALL (0.75) | 100 / SMALL (0.71) | 99 / SMALL (0.62) |
| Hyundai Santa Cruz | LARGE (4.94) | 298 | 6 | 2 | 83 / LARGE (0.45) | 50 / MID (0.25) x | 0 / SMALL (0.30) x | 0 / SMALL (0.25) x | 33 / MID (0.25) x | 17 / SMALL (0.27) x | 0 / SMALL (0.26) x |
| Hyundai Santa Fe | MID (4.56) | 565 | 35 | 13 | 66 / MID (0.43) | 66 / MID (0.51) | 17 / SMALL (0.30) x | 20 / SMALL (0.35) x | 69 / MID (0.46) | 0 / SMALL (0.17) x | 0 / SMALL (0.22) x |
| Jeep Grand Cherokee | SMALL (4.46) | 212 | 18 | 9 | 22 / LARGE (0.23) x | 39 / LARGE (0.37) x | 72 / SMALL (0.47) | 6 / LARGE (0.30) x | 89 / SMALL (0.51) | 100 / SMALL (0.78) | 94 / SMALL (0.60) |
| Kia Carnival | LARGE (5.30) | 1190 | 71 | 21 | 77 / LARGE (0.50) | 77 / LARGE (0.53) | 17 / SMALL (0.31) x | 8 / SMALL (0.24) x | 61 / LARGE (0.41) | 0 / SMALL (0.15) x | 0 / SMALL (0.18) x |
| Kia Ceed | SMALL (4.16) | 908 | 21 | 6 | 90 / SMALL (0.63) | 90 / SMALL (0.72) | 67 / SMALL (0.39) | 76 / SMALL (0.42) | 95 / SMALL (0.72) | 100 / SMALL (0.61) | 100 / SMALL (0.55) |
| Kia EV9 | LARGE (5.30) | 863 | 30 | 7 | 53 / LARGE (0.40) | 93 / LARGE (0.58) | 10 / SMALL (0.30) x | 7 / MID (0.28) x | 70 / LARGE (0.40) | 0 / SMALL (0.18) x | 0 / SMALL (0.26) x |
| Kia K5 | MID (4.61) | 115 | 4 | 3 | 25 / SMALL (0.40) x | 25 / MID (0.46) | 0 / SMALL (0.32) x | 25 / SMALL (0.35) x | 0 / SMALL (0.41) x | 0 / SMALL (0.20) x | 0 / SMALL (0.23) x |
| Kia Optima | MID (4.53) | 119 | 13 | 6 | 85 / MID (0.56) | 62 / MID (0.49) | 0 / SMALL (0.27) x | 0 / SMALL (0.34) x | 77 / MID (0.53) | 0 / SMALL (0.09) x | 0 / SMALL (0.16) x |
| Kia Sorento | MID (4.64) | 404 | 21 | 5 | 57 / SMALL (0.40) x | 14 / SMALL (0.29) x | 5 / SMALL (0.30) x | 33 / SMALL (0.35) x | 29 / SMALL (0.30) x | 0 / SMALL (0.12) x | 5 / SMALL (0.15) x |
| Kia Stinger | MID (4.51) | 542 | 12 | 4 | 0 / SMALL (0.33) x | 8 / SMALL (0.31) x | 0 / SMALL (0.28) x | 83 / MID (0.41) | 0 / SMALL (0.27) x | 0 / SMALL (0.10) x | 0 / SMALL (0.19) x |
| Lexus RX | MID (4.55) | 497 | 12 | 8 | 17 / SMALL (0.32) x | 25 / LARGE (0.34) x | 42 / SMALL (0.30) x | 0 / SMALL (0.32) x | 8 / SMALL (0.18) x | 0 / SMALL (0.19) x | 0 / SMALL (0.23) x |
| Nissan Altima | MID (4.56) | 144 | 6 | 3 | 0 / SMALL (0.20) x | 0 / SMALL (0.21) x | 0 / SMALL (0.22) x | 33 / SMALL (0.32) x | 0 / SMALL (0.17) x | 0 / SMALL (0.14) x | 17 / SMALL (0.24) x |
| Nissan Leaf | SMALL (4.16) | 945 | 56 | 15 | 61 / SMALL (0.56) | 95 / SMALL (0.69) | 93 / SMALL (0.50) | 91 / SMALL (0.56) | 100 / SMALL (0.77) | 100 / SMALL (0.77) | 98 / SMALL (0.61) |
| Tesla Model S | LARGE (4.97) | 374 | 18 | 10 | 6 / MID (0.09) x | 11 / MID (0.10) x | 17 / SMALL (0.28) x | 17 / SMALL (0.28) x | 0 / MID (0.05) x | 6 / SMALL (0.23) x | 0 / SMALL (0.25) x |
| Tesla Model Y | MID (4.73) | 1150 | 58 | 26 | 66 / MID (0.45) | 40 / SMALL (0.40) x | 16 / SMALL (0.29) x | 67 / MID (0.42) | 2 / SMALL (0.19) x | 0 / SMALL (0.17) x | 12 / SMALL (0.24) x |
| Toyota Avalon | MID (4.50) | 93 | 4 | 3 | 50 / SMALL (0.44) x | 25 / SMALL (0.34) x | 0 / SMALL (0.36) x | 0 / SMALL (0.34) x | 0 / SMALL (0.13) x | 0 / SMALL (0.19) x | 0 / SMALL (0.12) x |
| Toyota Corolla | SMALL (4.09) | 4560 | 164 | 44 | 95 / SMALL (0.72) | 93 / SMALL (0.65) | 60 / SMALL (0.41) | 80 / SMALL (0.51) | 99 / SMALL (0.79) | 98 / SMALL (0.61) | 99 / SMALL (0.61) |
| Toyota Sienna | LARGE (5.23) | 379 | 11 | 5 | 45 / LARGE (0.42) | 55 / LARGE (0.56) | 0 / SMALL (0.27) x | 0 / SMALL (0.20) x | 45 / SMALL (0.40) x | 9 / SMALL (0.20) x | 0 / SMALL (0.28) x |
| Toyota Yaris | SMALL (3.78) | 99 | 10 | 3 | 80 / SMALL (0.68) | 80 / SMALL (0.61) | 60 / SMALL (0.42) | 70 / SMALL (0.46) | 80 / SMALL (0.70) | 100 / SMALL (0.80) | 100 / SMALL (0.67) |
| Volkswagen Atlas | LARGE (5.11) | 515 | 20 | 6 | 0 / SMALL (0.14) x | 0 / SMALL (0.07) x | 5 / SMALL (0.25) x | 0 / SMALL (0.16) x | 0 / SMALL (0.12) x | 0 / SMALL (0.12) x | 0 / SMALL (0.18) x |
| Volkswagen ID.3 | SMALL (4.31) | 57 | 6 | 3 | 100 / SMALL (0.71) | 100 / SMALL (0.71) | 83 / SMALL (0.46) | 83 / SMALL (0.46) | 100 / SMALL (0.82) | 100 / SMALL (0.69) | 83 / SMALL (0.58) |
| Volkswagen ID.4 | SMALL (4.36) | 227 | 17 | 5 | 94 / SMALL (0.57) | 94 / SMALL (0.66) | 76 / SMALL (0.45) | 76 / SMALL (0.49) | 100 / SMALL (0.78) | 100 / SMALL (0.71) | 71 / SMALL (0.59) |
| Volkswagen Jetta | SMALL (4.14) | 431 | 28 | 5 | 82 / SMALL (0.65) | 82 / SMALL (0.67) | 93 / SMALL (0.48) | 79 / SMALL (0.49) | 96 / SMALL (0.81) | 100 / SMALL (0.75) | 100 / SMALL (0.60) |
| Volkswagen Passat | SMALL (4.41) | 2216 | 48 | 11 | 81 / SMALL (0.56) | 71 / SMALL (0.61) | 90 / SMALL (0.50) | 73 / SMALL (0.45) | 94 / SMALL (0.72) | 100 / SMALL (0.68) | 100 / SMALL (0.57) |

Model-level decisions correct per class (size):

| model | SMALL models correct | MID models correct | LARGE models correct | all |
|---|---|---|---|---|
| full | 14/15 | 6/14 | 8/12 | 28/41 |
| V1 lateral + speed | 13/15 | 5/14 | 7/12 | 25/41 |
| V2 longitudinal-only | 14/15 | 0/14 | 3/12 | 17/41 |
| V3 speed-only | 14/15 | 3/14 | 3/12 | 20/41 |
| full+V1 | 15/15 | 5/14 | 3/12 | 23/41 |
| full+V2 | 15/15 | 0/14 | 0/12 | 15/41 |
| full+V3 | 15/15 | 0/14 | 0/12 | 15/41 |

## bev_ice (BEV / ICE); cells = drive acc % / model-level decision (P true), x = miss

| held-out model | label (footprint) | win | drives | veh | full | V1 lateral + speed | V2 longitudinal-only | V3 speed-only | full+V1 | full+V2 | full+V3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Audi Q5 | ICE (4.53) | 101 | 9 | 3 | 100 / ICE (0.80) | 100 / ICE (0.82) | 78 / ICE (0.66) | 100 / ICE (0.68) | 100 / ICE (0.69) | 100 / ICE (0.74) | 100 / ICE (0.72) |
| Chevrolet Equinox | ICE (4.30) | 165 | 12 | 3 | 100 / ICE (0.84) | 100 / ICE (0.87) | 100 / ICE (0.70) | 92 / ICE (0.72) | 100 / ICE (0.71) | 100 / ICE (0.79) | 100 / ICE (0.75) |
| Chevrolet Trailblazer | ICE (4.10) | 102 | 8 | 3 | 100 / ICE (0.90) | 100 / ICE (0.87) | 100 / ICE (0.80) | 100 / ICE (0.74) | 100 / ICE (0.80) | 100 / ICE (0.86) | 100 / ICE (0.81) |
| Ford F-150 Lightning | BEV (6.41) | 352 | 22 | 3 | 86 / BEV (0.63) | 41 / BEV (0.52) | 95 / BEV (0.67) | 86 / BEV (0.61) | 14 / ICE (0.35) x | 18 / ICE (0.40) x | 23 / ICE (0.45) x |
| Ford Ranger | ICE (5.30) | 109 | 10 | 5 | 90 / ICE (0.70) | 90 / ICE (0.70) | 100 / ICE (0.70) | 100 / ICE (0.67) | 100 / ICE (0.76) | 100 / ICE (0.77) | 80 / ICE (0.65) |
| GMC Yukon | ICE (5.17) | 373 | 13 | 5 | 100 / ICE (0.84) | 100 / ICE (0.80) | 100 / ICE (0.73) | 62 / ICE (0.59) | 100 / ICE (0.82) | 100 / ICE (0.84) | 100 / ICE (0.72) |
| GWM Haval H6 | ICE (4.48) | 377 | 17 | 3 | 100 / ICE (0.78) | 76 / ICE (0.56) | 82 / ICE (0.72) | 82 / ICE (0.67) | 100 / ICE (0.73) | 100 / ICE (0.79) | 100 / ICE (0.69) |
| Genesis G80 | ICE (4.95) | 173 | 5 | 3 | 100 / ICE (0.73) | 20 / BEV (0.33) x | 100 / ICE (0.67) | 100 / ICE (0.65) | 100 / ICE (0.74) | 100 / ICE (0.79) | 100 / ICE (0.64) |
| Genesis G90 | ICE (5.21) | 346 | 9 | 3 | 100 / ICE (0.74) | 56 / ICE (0.53) | 100 / ICE (0.74) | 100 / ICE (0.63) | 78 / ICE (0.68) | 100 / ICE (0.82) | 100 / ICE (0.69) |
| Genesis GV60 | BEV (4.73) | 1391 | 69 | 3 | 99 / BEV (0.83) | 99 / BEV (0.77) | 100 / BEV (0.70) | 97 / BEV (0.70) | 83 / BEV (0.55) | 16 / ICE (0.43) x | 23 / ICE (0.44) x |
| Genesis GV70 | ICE (4.73) | 782 | 24 | 9 | 100 / ICE (0.73) | 33 / BEV (0.48) x | 100 / ICE (0.72) | 96 / ICE (0.63) | 83 / ICE (0.61) | 100 / ICE (0.77) | 83 / ICE (0.65) |
| Genesis GV80 | ICE (4.96) | 417 | 37 | 8 | 84 / ICE (0.73) | 68 / ICE (0.59) | 95 / ICE (0.75) | 95 / ICE (0.71) | 68 / ICE (0.58) | 100 / ICE (0.77) | 51 / ICE (0.58) |
| Honda HR-V | ICE (4.16) | 282 | 19 | 7 | 95 / ICE (0.84) | 95 / ICE (0.80) | 100 / ICE (0.81) | 100 / ICE (0.85) | 95 / ICE (0.76) | 95 / ICE (0.82) | 84 / ICE (0.75) |
| Hyundai Azera | ICE (4.57) | 74 | 4 | 3 | 75 / ICE (0.55) | 50 / ICE (0.57) | 100 / ICE (0.70) | 100 / ICE (0.81) | 75 / BEV (0.49) x | 100 / ICE (0.76) | 75 / ICE (0.70) |
| Hyundai Ioniq | BEV (4.21) | 219 | 10 | 6 | 90 / BEV (0.79) | 90 / BEV (0.64) | 70 / BEV (0.59) | 90 / BEV (0.59) | 40 / ICE (0.44) x | 10 / ICE (0.35) x | 10 / ICE (0.37) x |
| Hyundai Kona | ICE (4.03) | 2889 | 121 | 21 | 70 / ICE (0.67) | 68 / ICE (0.60) | 78 / ICE (0.72) | 72 / ICE (0.77) | 90 / ICE (0.73) | 93 / ICE (0.82) | 92 / ICE (0.76) |
| Hyundai Santa Cruz | ICE (4.94) | 298 | 6 | 2 | 100 / ICE (0.69) | 17 / ICE (0.51) | 83 / ICE (0.64) | 100 / ICE (0.68) | 83 / ICE (0.66) | 100 / ICE (0.73) | 100 / ICE (0.70) |
| Hyundai Santa Fe | ICE (4.56) | 373 | 21 | 7 | 95 / ICE (0.73) | 52 / ICE (0.54) | 86 / ICE (0.67) | 90 / ICE (0.73) | 76 / ICE (0.67) | 100 / ICE (0.74) | 100 / ICE (0.71) |
| Jeep Grand Cherokee | ICE (4.46) | 212 | 18 | 9 | 100 / ICE (0.93) | 94 / ICE (0.84) | 94 / ICE (0.83) | 100 / ICE (0.84) | 100 / ICE (0.71) | 100 / ICE (0.75) | 100 / ICE (0.81) |
| Kia Carnival | ICE (5.30) | 938 | 57 | 14 | 96 / ICE (0.82) | 44 / ICE (0.52) | 98 / ICE (0.77) | 95 / ICE (0.80) | 53 / ICE (0.62) | 98 / ICE (0.76) | 96 / ICE (0.72) |
| Kia Ceed | ICE (4.15) | 101 | 5 | 3 | 100 / ICE (0.80) | 80 / ICE (0.54) | 100 / ICE (0.63) | 100 / ICE (0.58) | 100 / ICE (0.66) | 100 / ICE (0.73) | 80 / ICE (0.76) |
| Kia EV9 | BEV (5.30) | 863 | 30 | 7 | 100 / BEV (0.75) | 93 / BEV (0.65) | 100 / BEV (0.65) | 100 / BEV (0.72) | 30 / ICE (0.38) x | 17 / ICE (0.32) x | 7 / ICE (0.36) x |
| Kia K5 | ICE (4.61) | 115 | 4 | 3 | 100 / ICE (0.75) | 75 / ICE (0.68) | 100 / ICE (0.73) | 100 / ICE (0.76) | 75 / ICE (0.71) | 100 / ICE (0.80) | 75 / ICE (0.66) |
| Kia Optima | ICE (4.53) | 25 | 3 | 3 | 100 / ICE (0.86) | 33 / ICE (0.54) | 100 / ICE (0.79) | 100 / ICE (0.78) | 67 / ICE (0.55) | 100 / ICE (0.64) | 100 / ICE (0.74) |
| Kia Sorento | ICE (4.64) | 34 | 2 | 2 | 50 / ICE (0.66) | 0 / BEV (0.43) x | 100 / ICE (0.58) | 100 / ICE (0.60) | 100 / ICE (0.76) | 100 / ICE (0.78) | 100 / ICE (0.63) |
| Kia Stinger | ICE (4.51) | 542 | 12 | 4 | 100 / ICE (0.77) | 58 / BEV (0.49) x | 100 / ICE (0.58) | 100 / ICE (0.62) | 100 / ICE (0.71) | 100 / ICE (0.67) | 100 / ICE (0.78) |
| Lexus RX | ICE (4.55) | 121 | 4 | 2 | 100 / ICE (0.67) | 100 / ICE (0.95) | 100 / ICE (0.68) | 100 / ICE (0.73) | 100 / ICE (0.79) | 100 / ICE (0.69) | 100 / ICE (0.73) |
| Nissan Altima | ICE (4.56) | 144 | 6 | 3 | 83 / ICE (0.63) | 83 / ICE (0.60) | 100 / ICE (0.63) | 100 / ICE (0.62) | 33 / BEV (0.45) x | 83 / ICE (0.62) | 100 / ICE (0.68) |
| Nissan Leaf | BEV (4.16) | 945 | 56 | 15 | 64 / BEV (0.60) | 16 / ICE (0.34) x | 21 / ICE (0.37) x | 18 / ICE (0.37) x | 9 / ICE (0.30) x | 5 / ICE (0.26) x | 0 / ICE (0.29) x |
| Tesla Model S | BEV (4.97) | 374 | 18 | 10 | 72 / ICE (0.46) x | 72 / ICE (0.39) x | 89 / BEV (0.66) | 94 / BEV (0.68) | 33 / ICE (0.29) x | 17 / ICE (0.37) x | 28 / ICE (0.37) x |
| Tesla Model Y | BEV (4.73) | 1150 | 58 | 26 | 100 / BEV (0.83) | 93 / BEV (0.71) | 93 / BEV (0.73) | 93 / BEV (0.73) | 53 / BEV (0.53) | 28 / ICE (0.43) x | 12 / ICE (0.40) x |
| Toyota Avalon | ICE (4.47) | 39 | 2 | 1 | 100 / ICE (0.92) | 100 / ICE (0.78) | 100 / ICE (0.72) | 100 / ICE (0.85) | 100 / ICE (0.62) | 100 / ICE (0.76) | 100 / ICE (0.82) |
| Toyota Corolla | ICE (4.10) | 1031 | 35 | 19 | 94 / ICE (0.72) | 100 / ICE (0.86) | 100 / ICE (0.74) | 97 / ICE (0.82) | 100 / ICE (0.79) | 100 / ICE (0.78) | 100 / ICE (0.75) |
| Toyota Sienna | ICE (5.21) | 308 | 8 | 3 | 100 / ICE (0.82) | 100 / ICE (0.94) | 100 / ICE (0.66) | 100 / ICE (0.66) | 100 / ICE (0.85) | 100 / ICE (0.68) | 100 / ICE (0.74) |
| Volkswagen Atlas | ICE (5.11) | 515 | 20 | 6 | 100 / ICE (0.89) | 100 / ICE (0.85) | 100 / ICE (0.80) | 100 / ICE (0.83) | 100 / ICE (0.69) | 100 / ICE (0.76) | 100 / ICE (0.76) |
| Volkswagen ID.3 | BEV (4.31) | 57 | 6 | 3 | 100 / BEV (0.63) | 50 / BEV (0.58) | 100 / BEV (0.69) | 100 / BEV (0.76) | 50 / BEV (0.52) | 17 / ICE (0.47) x | 50 / BEV (0.52) |
| Volkswagen ID.4 | BEV (4.36) | 227 | 17 | 5 | 88 / BEV (0.78) | 65 / BEV (0.62) | 82 / BEV (0.74) | 71 / BEV (0.73) | 88 / BEV (0.50) | 35 / ICE (0.47) x | 53 / ICE (0.45) x |
| Volkswagen Jetta | ICE (4.14) | 431 | 28 | 5 | 100 / ICE (0.88) | 100 / ICE (0.84) | 100 / ICE (0.83) | 96 / ICE (0.77) | 100 / ICE (0.81) | 100 / ICE (0.87) | 100 / ICE (0.81) |
| Volkswagen Passat | ICE (4.42) | 1300 | 31 | 4 | 100 / ICE (0.84) | 100 / ICE (0.88) | 100 / ICE (0.81) | 100 / ICE (0.75) | 100 / ICE (0.75) | 100 / ICE (0.83) | 100 / ICE (0.77) |

Model-level decisions correct per class (bev_ice):

| model | BEV models correct | ICE models correct | all |
|---|---|---|---|
| full | 8/9 | 30/30 | 38/39 |
| V1 lateral + speed | 7/9 | 26/30 | 33/39 |
| V2 longitudinal-only | 8/9 | 30/30 | 38/39 |
| V3 speed-only | 8/9 | 30/30 | 38/39 |
| full+V1 | 4/9 | 28/30 | 32/39 |
| full+V2 | 0/9 | 30/30 | 30/39 |
| full+V3 | 1/9 | 30/30 | 31/39 |

