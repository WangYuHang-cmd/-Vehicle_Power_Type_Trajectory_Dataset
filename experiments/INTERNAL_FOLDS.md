# Internal car-model-disjoint folds

Selection folds for the method study (hyper-parameters, early stopping, aggregation and decision rules are chosen on these, never on the test split). Pool = drives of `split_platform` in {train, validation}. A fold holds out whole car models; its training set is the rest of the pool minus every drive whose identity group also drove a held-out model, so an identity group never sits on both sides of a fold (the published splits keep test groups out of train and validation in the same way). Candidates: >= 2 identity groups, >= 100 eligible windows outside fleet groups, <= 35 % of their size x powertrain cell's pool windows (so a cell's dominant model, e.g. the pickup / SUV pair that carries most LARGE hours or the compact that carries most SMALL hours, is never held out). Folds are balanced over the six size x {BEV, ICE} cells by validated hours (hybrids and unknown powertrains count with the combustion bucket) and each holds >= 2 models whose footprint lies within 0.15 m^2 of a size threshold (`near` column). Hours = eligible pool windows x 60 s; validated hours leave out the fleet groups' windows.

**Fleet rule.** One identity group of the pool is a shared device pool (the lab fleet: 19 car models, 33 vehicles, about 480 h, every drive in the training split), not an owner whose habits could stand in for a car model. Under the literal group rule every fold holds out some of its car models, so the whole group leaves the training set of every fold and 13-17 % of the pool windows are never trained on (100 % of them from this one group), while the fold validation sets still contain its drives: a population shift inside every fold and a fold training population unlike the final full-train run. Groups spanning >= 5 car models are therefore pinned to the training side: their drives of held-out models are dropped from the fold (neither trained on nor validated; they train in the other folds), their other drives stay in training. Validation keeps only drives of the other identity groups, which is what the test split holds as well. `internal_folds.py --fleet-min-models 0` rebuilds the literal rule.

## Task `size`

Pool: 168,588 eligible windows (2810 h) from 117 car models; 57 candidate models; seed 0; 3 folds.

Fleet groups in the pool: 1 (>= 5 car models; 1370 drives, 22,468 windows = 13.3 % of the pool). Pool windows never trained on in any fold: 0 (0.0 %), against 22,468 (13.3 %, 22,468 of them in fleet groups) under the literal group rule; pool windows in no fold at all (neither trained on nor validated anywhere): 0.

| fold | held-out models | hours (validated) | drives (val / train / dropped) | windows (val / train / dropped, of which fleet) | near-threshold models | validated hours by cell |
|---|---|---|---|---|---|---|
| 0 | 19 | 490 (471) | 1146 / 6045 / 394 | 28,258 / 133,386 / 6,944, 1,161 | 9 | LARGE / ICE 110, MID / BEV 89, MID / ICE 69, SMALL / BEV 53, SMALL / ICE 151 |
| 1 | 19 | 511 (455) | 1045 / 6254 / 286 | 27,313 / 134,850 / 6,425, 3,357 | 9 | LARGE / ICE 107, MID / BEV 88, MID / ICE 69, SMALL / BEV 40, SMALL / ICE 151 |
| 2 | 19 | 562 (431) | 1108 / 5578 / 899 | 25,869 / 125,022 / 17,697, 7,842 | 9 | LARGE / ICE 108, MID / BEV 77, MID / ICE 69, SMALL / BEV 26, SMALL / ICE 151 |

### Fold 0

| car model | cell | hours | validated hours | groups | drives | footprint m^2 | near |
|---|---|---|---|---|---|---|---|
| RAM 2500/3500 HD | LARGE / ICE | 57.7 | 57.7 | 8 | 26 | 6.57 |  |
| Hyundai Palisade | LARGE / ICE | 37.1 | 37.1 | 13 | 75 | 4.96 | yes |
| Honda Ridgeline | LARGE / ICE | 10.1 | 10.1 | 9 | 23 | 5.33 |  |
| Ford Expedition | LARGE / ICE | 4.7 | 4.7 | 2 | 15 | 5.53 |  |
| Hyundai Ioniq 6 | MID / BEV | 89.0 | 89.0 | 13 | 237 | 4.80 | yes |
| Honda Accord | MID / ICE | 58.9 | 49.2 | 23 | 104 | 4.54 | yes |
| Acura RDX | MID / ICE | 20.8 | 13.7 | 7 | 75 | 4.50 | yes |
| Volvo V60 | MID / ICE | 5.7 | 5.7 | 2 | 13 | 4.62 |  |
| Chevrolet Bolt EV | SMALL / BEV | 52.9 | 52.9 | 9 | 124 | 3.90 |  |
| Volkswagen Golf | SMALL / ICE | 36.7 | 36.7 | 17 | 119 | 4.02 |  |
| Honda CR-V | SMALL / ICE | 33.4 | 33.4 | 28 | 93 | 4.37 | yes |
| Subaru Legacy | SMALL / ICE | 19.4 | 19.4 | 3 | 56 | 4.37 | yes |
| Subaru Outback | SMALL / ICE | 18.3 | 18.3 | 15 | 31 | 4.27 |  |
| Kia Sportage | SMALL / ICE | 12.1 | 12.1 | 10 | 46 | 4.42 | yes |
| Mazda CX-5 | SMALL / ICE | 9.2 | 9.2 | 10 | 29 | 4.31 | yes |
| Honda Insight | SMALL / ICE | 7.3 | 7.3 | 4 | 15 | 4.20 |  |
| Acura Integra | SMALL / ICE | 9.3 | 6.8 | 5 | 35 | 4.24 |  |
| Ford Bronco Sport | SMALL / ICE | 3.9 | 3.9 | 2 | 8 | 4.28 |  |
| Skoda Kodiaq | SMALL / ICE | 3.7 | 3.7 | 2 | 4 | 4.41 | yes |

### Fold 1

| car model | cell | hours | validated hours | groups | drives | footprint m^2 | near |
|---|---|---|---|---|---|---|---|
| Chevrolet Silverado 1500 | LARGE / ICE | 91.7 | 91.7 | 5 | 194 | 6.65 |  |
| Ford Maverick | LARGE / ICE | 15.8 | 13.8 | 7 | 36 | 4.93 | yes |
| Mazda CX-9 | LARGE / ICE | 1.7 | 1.7 | 2 | 8 | 5.15 |  |
| Kia EV6 | MID / BEV | 53.4 | 49.1 | 31 | 125 | 4.74 | yes |
| Tesla Model 3 | MID / BEV | 62.2 | 38.7 | 20 | 191 | 4.57 | yes |
| Genesis G70 | MID / ICE | 23.8 | 23.8 | 3 | 38 | 4.54 | yes |
| Acura TLX | MID / ICE | 16.0 | 14.2 | 5 | 30 | 4.44 | yes |
| Toyota Camry | MID / ICE | 26.2 | 13.7 | 13 | 60 | 4.47 | yes |
| Toyota Highlander | MID / ICE | 11.8 | 11.8 | 14 | 25 | 4.68 |  |
| Volkswagen Sharan | MID / ICE | 5.3 | 5.3 | 2 | 19 | 4.65 |  |
| Chevrolet Bolt EUV | SMALL / BEV | 40.5 | 40.5 | 17 | 70 | 3.99 |  |
| Toyota RAV4 | SMALL / ICE | 107.2 | 95.5 | 69 | 191 | 4.34 | yes |
| Subaru Forester | SMALL / ICE | 23.9 | 23.9 | 13 | 57 | 4.19 |  |
| Toyota C-HR | SMALL / ICE | 15.3 | 15.3 | 3 | 8 | 4.08 |  |
| Honda City | SMALL / ICE | 5.3 | 5.3 | 3 | 15 | 3.86 |  |
| Lexus IS | SMALL / ICE | 4.9 | 4.9 | 4 | 28 | 4.32 | yes |
| Kia Forte | SMALL / ICE | 2.4 | 2.4 | 2 | 3 | 4.37 | yes |
| Audi A3 | SMALL / ICE | 2.1 | 2.1 | 3 | 3 | 3.97 |  |
| Kia Seltos | SMALL / ICE | 1.8 | 1.8 | 2 | 5 | 4.16 |  |

### Fold 2

| car model | cell | hours | validated hours | groups | drives | footprint m^2 | near |
|---|---|---|---|---|---|---|---|
| Honda Odyssey | LARGE / ICE | 66.0 | 66.0 | 15 | 134 | 5.14 |  |
| Ford Explorer | LARGE / ICE | 14.7 | 14.7 | 7 | 14 | 5.14 |  |
| Honda Pilot | LARGE / ICE | 14.4 | 13.9 | 18 | 43 | 4.97 | yes |
| RAM 1500 | LARGE / ICE | 13.8 | 13.8 | 14 | 44 | 6.73 |  |
| Hyundai Ioniq 5 | MID / BEV | 84.5 | 49.4 | 37 | 216 | 4.83 | yes |
| Ford Mustang Mach-E | MID / BEV | 40.2 | 27.4 | 8 | 68 | 4.84 | yes |
| Acura MDX | MID / ICE | 111.4 | 35.2 | 5 | 414 | 4.75 | yes |
| Skoda Superb | MID / ICE | 14.1 | 14.1 | 4 | 18 | 4.48 | yes |
| Subaru Ascent | MID / ICE | 13.1 | 13.1 | 8 | 37 | 4.72 |  |
| Lexus ES | MID / ICE | 6.6 | 6.6 | 5 | 12 | 4.61 |  |
| Cupra Born | SMALL / BEV | 24.7 | 24.7 | 2 | 47 | 4.22 |  |
| BYD Atto 3 | SMALL / BEV | 1.7 | 1.7 | 2 | 4 | 4.29 | yes |
| Hyundai Elantra | SMALL / ICE | 92.5 | 92.5 | 17 | 232 | 4.31 | yes |
| Subaru Impreza | SMALL / ICE | 20.9 | 20.9 | 19 | 40 | 4.13 |  |
| Subaru Crosstrek | SMALL / ICE | 13.5 | 13.5 | 8 | 44 | 4.15 |  |
| Volkswagen Tiguan | SMALL / ICE | 15.7 | 9.6 | 10 | 33 | 4.33 | yes |
| Volkswagen Taos | SMALL / ICE | 5.3 | 5.3 | 2 | 15 | 4.18 |  |
| Lexus NX | SMALL / ICE | 4.5 | 4.5 | 2 | 3 | 4.20 |  |
| Jeep Cherokee | SMALL / ICE | 4.2 | 4.2 | 2 | 12 | 4.30 | yes |

### Never held out (not a candidate)

| car model | cell | hours | groups | windows (outside fleet groups) | cell share |
|---|---|---|---|---|---|
| Rivian R1T/R1S | LARGE / BEV | 361.7 | 42 | 21,700 (21,700) | 95 % |
| Honda Civic | SMALL / ICE | 243.8 | 39 | 14,625 (6,476) | 46 % |
| Kia Niro | SMALL / BEV | 135.5 | 29 | 8,131 (7,146) | 42 % |
| Toyota Prius | SMALL / HEV | 96.3 | 37 | 5,776 (5,776) | 45 % |
| Chevrolet Volt | SMALL / PHEV | 84.7 | 14 | 5,080 (5,080) | 60 % |
| Hyundai Tucson | MID / NA | 67.6 | 14 | 4,056 (4,056) | 64 % |
| Ford Escape | SMALL / NA | 67.1 | 8 | 4,025 (4,025) | 49 % |
| Hyundai Sonata | MID / HEV | 63.2 | 25 | 3,790 (3,790) | 45 % |
| Ford F-150 | LARGE / NA | 57.9 | 11 | 3,473 (3,473) | 65 % |
| Tesla Model X | LARGE / BEV | 14.2 | 2 | 851 (32) | 4 % |
| Chrysler Pacifica | LARGE / PHEV | 9.3 | 13 | 558 (558) | 100 % |
| Polestar 2 | SMALL / BEV | 4.1 | 1 | 246 (246) | 2 % |
| Acura ADX | SMALL / ICE | 3.0 | 2 | 182 (28) | 1 % |
| Cadillac XT6 | MID / ICE | 2.3 | 1 | 139 (139) | 1 % |
| Hyundai Genesis | LARGE / ICE | 2.2 | 1 | 130 (130) | 1 % |
| Porsche Macan | MID / ICE | 2.1 | 1 | 128 (128) | 1 % |
| Buick LaCrosse | MID / ICE | 2.1 | 1 | 125 (125) | 1 % |
| Hyundai Ioniq 9 | LARGE / BEV | 1.7 | 1 | 103 (103) | 0 % |
| Ford Focus | SMALL / ICE | 1.6 | 2 | 94 (94) | 0 % |
| Subaru BRZ | SMALL / ICE | 1.4 | 1 | 86 (86) | 0 % |
| Peugeot 508 | MID / ICE | 1.4 | 1 | 85 (85) | 1 % |
| Kia K8 | MID / HEV | 1.4 | 1 | 83 (83) | 1 % |
| Lexus LS | LARGE / ICE | 1.3 | 1 | 80 (80) | 0 % |
| Nissan Rogue | SMALL / ICE | 1.3 | 1 | 79 (79) | 0 % |
| Cadillac XT4 | MID / ICE | 1.2 | 1 | 73 (73) | 0 % |
| ... 35 more models with fewer hours | | | | | |

## Task `bev_ice`

Pool: 121,381 eligible windows (2023 h) from 107 car models; 51 candidate models; seed 0; 3 folds.

Fleet groups in the pool: 1 (>= 5 car models; 1370 drives, 20,747 windows = 17.1 % of the pool). Pool windows never trained on in any fold: 0 (0.0 %), against 20,747 (17.1 %, 20,747 of them in fleet groups) under the literal group rule; pool windows in no fold at all (neither trained on nor validated anywhere): 0.

| fold | held-out models | hours (validated) | drives (val / train / dropped) | windows (val / train / dropped, of which fleet) | near-threshold models | validated hours by cell |
|---|---|---|---|---|---|---|
| 0 | 16 | 368 (368) | 859 / 6618 / 108 | 22,081 / 97,977 / 1,323, 0 | 7 | LARGE / ICE 98, MID / BEV 89, MID / ICE 55, SMALL / BEV 40, SMALL / ICE 85 |
| 1 | 15 | 377 (339) | 1236 / 5871 / 478 | 20,353 / 97,193 / 3,835, 2,265 | 8 | LARGE / ICE 102, MID / BEV 77, MID / ICE 50, SMALL / BEV 26, SMALL / ICE 85 |
| 2 | 20 | 508 (381) | 1182 / 5711 / 692 | 22,851 / 89,338 / 9,192, 7,619 | 8 | LARGE / ICE 100, MID / BEV 88, MID / ICE 54, SMALL / BEV 53, SMALL / ICE 86 |

### Fold 0

| car model | cell | hours | validated hours | groups | drives | footprint m^2 | near |
|---|---|---|---|---|---|---|---|
| RAM 2500/3500 HD | LARGE / ICE | 57.7 | 57.7 | 8 | 26 | 6.57 |  |
| Hyundai Palisade | LARGE / ICE | 36.8 | 36.8 | 11 | 71 | 4.96 | yes |
| Chrysler Pacifica | LARGE / ICE | 3.9 | 3.9 | 5 | 12 | 5.36 |  |
| Hyundai Ioniq 6 | MID / BEV | 89.0 | 89.0 | 13 | 237 | 4.80 | yes |
| Genesis G70 | MID / ICE | 23.8 | 23.8 | 3 | 38 | 4.54 | yes |
| Skoda Superb | MID / ICE | 14.1 | 14.1 | 4 | 18 | 4.48 | yes |
| Honda Accord | MID / ICE | 6.0 | 6.0 | 6 | 16 | 4.54 | yes |
| Volvo V60 | MID / ICE | 5.7 | 5.7 | 2 | 13 | 4.62 |  |
| Lexus ES | MID / ICE | 5.3 | 5.3 | 4 | 9 | 4.61 |  |
| Chevrolet Bolt EUV | SMALL / BEV | 40.5 | 40.5 | 17 | 70 | 3.99 |  |
| Honda CR-V | SMALL / ICE | 32.5 | 32.5 | 26 | 87 | 4.37 | yes |
| Subaru Impreza | SMALL / ICE | 20.9 | 20.9 | 19 | 40 | 4.13 |  |
| Subaru Outback | SMALL / ICE | 18.3 | 18.3 | 15 | 31 | 4.27 |  |
| Volkswagen Taos | SMALL / ICE | 5.3 | 5.3 | 2 | 15 | 4.18 |  |
| Jeep Cherokee | SMALL / ICE | 4.2 | 4.2 | 2 | 12 | 4.30 | yes |
| Ford Bronco Sport | SMALL / ICE | 3.9 | 3.9 | 2 | 8 | 4.28 |  |

### Fold 1

| car model | cell | hours | validated hours | groups | drives | footprint m^2 | near |
|---|---|---|---|---|---|---|---|
| Chevrolet Silverado 1500 | LARGE / ICE | 91.7 | 91.7 | 5 | 194 | 6.65 |  |
| Honda Ridgeline | LARGE / ICE | 10.1 | 10.1 | 9 | 23 | 5.33 |  |
| Kia EV6 | MID / BEV | 53.4 | 49.1 | 31 | 125 | 4.74 | yes |
| Ford Mustang Mach-E | MID / BEV | 40.2 | 27.4 | 8 | 68 | 4.84 | yes |
| Hyundai Sonata | MID / ICE | 15.5 | 15.5 | 13 | 25 | 4.62 |  |
| Acura TLX | MID / ICE | 16.0 | 14.2 | 5 | 30 | 4.44 | yes |
| Acura RDX | MID / ICE | 20.8 | 13.7 | 7 | 75 | 4.50 | yes |
| Toyota Highlander | MID / ICE | 6.6 | 6.6 | 7 | 12 | 4.60 |  |
| Cupra Born | SMALL / BEV | 24.7 | 24.7 | 2 | 47 | 4.22 |  |
| BYD Atto 3 | SMALL / BEV | 1.7 | 1.7 | 2 | 4 | 4.29 | yes |
| Hyundai Elantra | SMALL / ICE | 39.1 | 39.1 | 11 | 75 | 4.31 | yes |
| Toyota RAV4 | SMALL / ICE | 43.2 | 31.5 | 18 | 88 | 4.34 | yes |
| Volkswagen Golf | SMALL / ICE | 5.8 | 5.8 | 6 | 13 | 4.02 |  |
| Lexus NX | SMALL / ICE | 4.5 | 4.5 | 2 | 3 | 4.20 |  |
| Skoda Kodiaq | SMALL / ICE | 3.7 | 3.7 | 2 | 4 | 4.41 | yes |

### Fold 2

| car model | cell | hours | validated hours | groups | drives | footprint m^2 | near |
|---|---|---|---|---|---|---|---|
| Honda Odyssey | LARGE / ICE | 66.0 | 66.0 | 15 | 134 | 5.14 |  |
| Honda Pilot | LARGE / ICE | 14.4 | 13.9 | 18 | 43 | 4.97 | yes |
| RAM 1500 | LARGE / ICE | 13.8 | 13.8 | 14 | 44 | 6.73 |  |
| Ford Expedition | LARGE / ICE | 4.7 | 4.7 | 2 | 15 | 5.53 |  |
| Mazda CX-9 | LARGE / ICE | 1.7 | 1.7 | 2 | 8 | 5.15 |  |
| Hyundai Ioniq 5 | MID / BEV | 84.5 | 49.4 | 37 | 216 | 4.83 | yes |
| Tesla Model 3 | MID / BEV | 62.2 | 38.7 | 20 | 191 | 4.57 | yes |
| Acura MDX | MID / ICE | 100.5 | 35.2 | 5 | 364 | 4.75 | yes |
| Subaru Ascent | MID / ICE | 13.1 | 13.1 | 8 | 37 | 4.72 |  |
| Volkswagen Sharan | MID / ICE | 5.3 | 5.3 | 2 | 19 | 4.65 |  |
| Chevrolet Bolt EV | SMALL / BEV | 52.9 | 52.9 | 9 | 124 | 3.90 |  |
| Subaru Forester | SMALL / ICE | 23.5 | 23.5 | 14 | 55 | 4.19 |  |
| Subaru Legacy | SMALL / ICE | 19.4 | 19.4 | 3 | 56 | 4.37 | yes |
| Subaru Crosstrek | SMALL / ICE | 13.0 | 13.0 | 7 | 40 | 4.15 |  |
| Mazda CX-5 | SMALL / ICE | 9.2 | 9.2 | 10 | 29 | 4.31 | yes |
| Acura Integra | SMALL / ICE | 9.3 | 6.8 | 5 | 35 | 4.24 |  |
| Honda City | SMALL / ICE | 5.3 | 5.3 | 3 | 15 | 3.86 |  |
| Lexus IS | SMALL / ICE | 4.9 | 4.9 | 4 | 28 | 4.32 | yes |
| Kia Forte | SMALL / ICE | 2.4 | 2.4 | 2 | 3 | 4.37 | yes |
| Kia Seltos | SMALL / ICE | 1.8 | 1.8 | 2 | 5 | 4.16 |  |

### Never held out (not a candidate)

| car model | cell | hours | groups | windows (outside fleet groups) | cell share |
|---|---|---|---|---|---|
| Rivian R1T/R1S | LARGE / BEV | 361.7 | 42 | 21,700 (21,700) | 95 % |
| Honda Civic | SMALL / ICE | 243.8 | 39 | 14,625 (6,476) | 46 % |
| Kia Niro | SMALL / BEV | 91.6 | 19 | 5,498 (4,513) | 42 % |
| Tesla Model X | LARGE / BEV | 14.2 | 2 | 851 (32) | 4 % |
| Toyota Camry | MID / ICE | 13.5 | 4 | 811 (56) | 6 % |
| Polestar 2 | SMALL / BEV | 4.1 | 1 | 246 (246) | 2 % |
| Acura ADX | SMALL / ICE | 3.0 | 2 | 182 (28) | 1 % |
| Hyundai Mufasa | NA / ICE | 2.5 | 1 | 147 (147) | 89 % |
| Cadillac XT6 | MID / ICE | 2.3 | 1 | 139 (139) | 1 % |
| Hyundai Genesis | LARGE / ICE | 2.2 | 1 | 130 (130) | 1 % |
| Porsche Macan | MID / ICE | 2.1 | 1 | 128 (128) | 1 % |
| Buick LaCrosse | MID / ICE | 2.1 | 1 | 125 (125) | 1 % |
| Hyundai Ioniq 9 | LARGE / BEV | 1.7 | 1 | 103 (103) | 0 % |
| Ford Focus | SMALL / ICE | 1.6 | 2 | 94 (94) | 0 % |
| Subaru BRZ | SMALL / ICE | 1.4 | 1 | 86 (86) | 0 % |
| Peugeot 508 | MID / ICE | 1.4 | 1 | 85 (85) | 1 % |
| Lexus LS | LARGE / ICE | 1.3 | 1 | 80 (80) | 0 % |
| Nissan Rogue | SMALL / ICE | 1.3 | 1 | 79 (79) | 0 % |
| Cadillac XT4 | MID / ICE | 1.2 | 1 | 73 (73) | 0 % |
| Volkswagen Caddy | SMALL / ICE | 1.2 | 1 | 70 (70) | 0 % |
| Honda Passport | LARGE / ICE | 1.1 | 2 | 63 (63) | 0 % |
| Ford Edge | MID / ICE | 1.0 | 1 | 60 (60) | 0 % |
| Cadillac Escalade ESV | LARGE / ICE | 0.9 | 1 | 54 (54) | 0 % |
| Chevrolet Malibu | MID / ICE | 0.8 | 2 | 50 (50) | 0 % |
| Luxgen n7 | MID / BEV | 0.8 | 1 | 50 (50) | 0 % |
| ... 31 more models with fewer hours | | | | | |
