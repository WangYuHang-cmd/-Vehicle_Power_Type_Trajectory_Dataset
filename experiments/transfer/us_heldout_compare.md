# Held-out models of `ccs` against `ccs_us` (full size model; us_heldout_compare.py)

Device = mean probability over the device's labelled windows -> arg-max; model = mean probability over the model's windows -> arg-max; truth = majority label; `internal` = the lab's own test windows of that model (internal_per_model.md, column `full`).

| held-out model | label | devices | windows | devices right (acc %) before -> after | delta (points) | model decision (P true) before -> after | flip | internal: drive acc / decision (P true) |
|---|---|---|---|---|---|---|---|---|
| Chevrolet Equinox | SMALL | 5 | 243 | 5/5 (100.0) -> 5/5 (100.0) | +0.0 | SMALL (0.64) -> SMALL (0.63) OK | same | 50 / SMALL (0.37) |
| Ford F-150 Lightning | LARGE | 6 | 113 | 5/6 (83.3) -> 6/6 (100.0) | +16.7 | LARGE (0.55) -> LARGE (0.75) OK | same | 95 / LARGE (0.73) |
| Ford Ranger | LARGE | 3 | 82 | 2/3 (66.7) -> 3/3 (100.0) | +33.3 | SMALL (0.38) -> LARGE (0.55) OK | recovered | 90 / LARGE (0.74) |
| Genesis G80 | LARGE | 7 | 304 | 1/7 (14.3) -> 7/7 (100.0) | +85.7 | SMALL (0.26) -> LARGE (0.46) OK | recovered | 0 / MID (0.13) miss |
| Genesis G90 | LARGE | 9 | 249 | 0/9 (0.0) -> 0/9 (0.0) | +0.0 | SMALL (0.06) -> SMALL (0.11) miss | same | 0 / MID (0.13) miss |
| Genesis GV60 | MID | 2 | 97 | 1/2 (50.0) -> 1/2 (50.0) | +0.0 | MID (0.52) -> MID (0.46) OK | same | 97 / MID (0.61) |
| Genesis GV70 | MID | 8 | 200 | 1/8 (12.5) -> 3/8 (37.5) | +25.0 | SMALL (0.36) -> SMALL (0.39) miss | same | 67 / MID (0.45) |
| Genesis GV80 | LARGE | 3 | 146 | 0/3 (0.0) -> 0/3 (0.0) | +0.0 | SMALL (0.06) -> SMALL (0.12) miss | same | 62 / LARGE (0.51) |
| Hyundai Azera | MID | 1 | 5 | 0/1 (0.0) -> 0/1 (0.0) | +0.0 | SMALL (0.32) -> SMALL (0.30) miss | same | 80 / MID (0.50) |
| Hyundai Ioniq | SMALL | 26 | 935 | 26/26 (100.0) -> 26/26 (100.0) | +0.0 | SMALL (0.70) -> SMALL (0.71) OK | same | 69 / SMALL (0.59) |
| Hyundai Kona | SMALL | 20 | 538 | 18/20 (90.0) -> 19/20 (95.0) | +5.0 | SMALL (0.63) -> SMALL (0.64) OK | same | 86 / SMALL (0.60) |
| Hyundai Santa Cruz | LARGE | 1 | 49 | 0/1 (0.0) -> 0/1 (0.0) | +0.0 | SMALL (0.09) -> SMALL (0.17) miss | same | 83 / LARGE (0.45) |
| Hyundai Santa Fe | MID | 62 | 2355 | 39/62 (62.9) -> 48/62 (77.4) | +14.5 | MID (0.46) -> MID (0.48) OK | same | 66 / MID (0.43) |
| Jeep Grand Cherokee | SMALL | 36 | 1438 | 26/36 (72.2) -> 11/36 (30.6) | -41.7 | SMALL (0.49) -> MID (0.35) miss | lost | 22 / LARGE (0.23) miss |
| Kia Carnival | LARGE | 15 | 477 | 0/15 (0.0) -> 0/15 (0.0) | +0.0 | SMALL (0.07) -> MID (0.12) miss | wrong->wrong | 77 / LARGE (0.50) |
| Kia K5 | MID | 3 | 99 | 0/3 (0.0) -> 0/3 (0.0) | +0.0 | SMALL (0.26) -> SMALL (0.30) miss | same | 25 / SMALL (0.40) miss |
| Kia Optima | MID | 5 | 163 | 0/5 (0.0) -> 0/5 (0.0) | +0.0 | SMALL (0.30) -> SMALL (0.25) miss | same | 85 / MID (0.56) |
| Kia Sorento | MID | 17 | 545 | 0/17 (0.0) -> 1/17 (5.9) | +5.9 | SMALL (0.26) -> SMALL (0.25) miss | same | 57 / SMALL (0.40) miss |
| Kia Stinger | MID | 10 | 479 | 10/10 (100.0) -> 10/10 (100.0) | +0.0 | MID (0.56) -> MID (0.56) OK | same | 0 / SMALL (0.33) miss |
| Lexus RX | MID | 124 | 1931 | 7/124 (5.6) -> 23/124 (18.5) | +12.9 | SMALL (0.32) -> SMALL (0.37) miss | same | 17 / SMALL (0.32) miss |
| Nissan Altima | MID | 4 | 193 | 0/4 (0.0) -> 0/4 (0.0) | +0.0 | SMALL (0.32) -> SMALL (0.27) miss | same | 0 / SMALL (0.20) miss |
| Nissan Leaf | SMALL | 28 | 916 | 17/28 (60.7) -> 16/28 (57.1) | -3.6 | SMALL (0.54) -> SMALL (0.50) OK | same | 61 / SMALL (0.56) |
| Tesla Model Y | MID | 21 | 538 | 17/21 (81.0) -> 14/21 (66.7) | -14.3 | MID (0.59) -> MID (0.51) OK | same | 66 / MID (0.45) |
| Toyota Avalon | MID | 37 | 1250 | 1/37 (2.7) -> 3/37 (8.1) | +5.4 | SMALL (0.29) -> SMALL (0.31) miss | same | 50 / SMALL (0.44) miss |
| Toyota Corolla | SMALL | 272 | 1952 | 259/272 (95.2) -> 262/272 (96.3) | +1.1 | SMALL (0.65) -> SMALL (0.64) OK | same | 95 / SMALL (0.72) |
| Toyota Sienna | LARGE | 94 | 1854 | 15/94 (16.0) -> 28/94 (29.8) | +13.8 | SMALL (0.19) -> SMALL (0.30) miss | same | 45 / LARGE (0.42) |
| Volkswagen Atlas | LARGE | 32 | 961 | 0/32 (0.0) -> 1/32 (3.1) | +3.1 | SMALL (0.09) -> SMALL (0.19) miss | same | 0 / SMALL (0.14) miss |
| Volkswagen Jetta | SMALL | 12 | 342 | 12/12 (100.0) -> 12/12 (100.0) | +0.0 | SMALL (0.72) -> SMALL (0.65) OK | same | 82 / SMALL (0.65) |
| Volkswagen Passat | SMALL | 8 | 256 | 8/8 (100.0) -> 8/8 (100.0) | +0.0 | SMALL (0.68) -> SMALL (0.59) OK | same | 81 / SMALL (0.56) |

Held-out stratum, full size model, `ccs` -> `ccs_us`: 29 models; devices right 470 -> 507 of 871 (54.0 -> 58.2 %); models right 13 -> 14; recovered 2, lost 1.
