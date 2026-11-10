# commaCarSegments: owner download selection (9 Nov 2026)

Built by `experiments/transfer/ccs_select.py` from the dataset index (`database.json`, 188,883 segments in 230 keys). Legacy keys folded with opendbc MIGRATION (0.3.1 (pip), 279 entries, 245 platforms) -> 185 platforms. Dropped: MOCK and fork keys (122 segments). The 464 devices shared with the lab's raw library were removed (18,158 segments), leaving 170,603 segments.

Selection rule: 'seen' models (models with training / validation drives in the lab's `split_platform` scheme) keep at most 400 segments per platform from at least 20 devices, taken round-robin as pieces of up to 5 consecutive segments of one route (multi-segment pieces before singletons, devices with consecutive runs first); held-out models (the 41 test models) and models absent from the lab inventory keep every segment.

**Total: 60,619 segments from 4,334 devices and 29,637 routes; about 95.8 GB at 1.58 MB per segment (index-wide mean), upper estimate 119.4 GB at 1.97 MB (mean of the five verification samples).**

The list itself (`select.txt`, one `<dongle>/<route>/<seg>` per line) stays on the data disk next to the index; it carries device ids and is never copied into the repository or into any markdown.

## Strata

| stratum | platforms | models | segments | devices | routes | ~GB |
|---|---|---|---|---|---|---|
| seen | 108 | 72 | 28,650 | 3,524 | 12,687 | 45.3 |
| held-out | 49 | 30 | 30,476 | 880 | 16,252 | 48.2 |
| absent | 15 | 15 | 1,493 | 37 | 698 | 2.4 |

## Per platform

Columns: index = segments / devices in the index; shared = segments removed with the lab-shared devices; selected = segments / devices / routes in the list; runs = share of the selected segments that have a selected neighbour segment in the same route (several windows per drive). Footprint and class from the lab table (`lab`) or the public spec table `specs_external.csv` (`public`); `bl` = borderline (|footprint - cut| < 0.05 m^2).

| platform | model | stratum | footprint | class | bl | powertrain | label | index seg / dev | shared | selected seg / dev / routes | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CHEVROLET_BOLT_EUV | Chevrolet Bolt EUV | seen | 3.986 | SMALL |  | BEV | lab | 7,664 / 176 | 511 | 400 / 165 / 167 | 0.90 |
| CHEVROLET_SILVERADO | Chevrolet Silverado 1500 | seen | 6.649 | LARGE |  | ICE | lab | 2,473 / 56 | 166 | 400 / 52 / 139 | 0.86 |
| CHRYSLER_PACIFICA_2018_HYBRID | Chrysler Pacifica | seen | 5.361 | LARGE |  | PHEV | lab | 1,387 / 32 | 72 | 400 / 30 / 190 | 0.56 |
| CHRYSLER_PACIFICA_2019_HYBRID | Chrysler Pacifica | seen | 5.361 | LARGE |  | PHEV | lab | 3,905 / 100 | 145 | 400 / 96 / 161 | 0.83 |
| CHRYSLER_PACIFICA_2020 | Chrysler Pacifica | seen | 5.361 | LARGE |  | ICE | lab | 1,823 / 41 | 0 | 400 / 41 / 176 | 0.69 |
| FORD_EXPLORER_MK6 | Ford Explorer | seen | 5.139 | LARGE |  | unknown (MIXED) | lab | 1,511 / 49 | 183 | 400 / 45 / 174 | 0.61 |
| FORD_F_150_MK14 | Ford F-150 | seen | 6.439 | LARGE |  | unknown (MIXED) | lab | 1,354 / 45 | 101 | 400 / 42 / 117 | 0.85 |
| FORD_MAVERICK_MK1 | Ford Maverick | seen | 4.929 | LARGE | yes | unknown (MIXED) | lab | 1,169 / 35 | 69 | 400 / 32 / 163 | 0.60 |
| HONDA_ACCORD | Honda Accord | seen | 4.542 | MID |  | HEV | lab | 4,568 / 129 | 267 | 400 / 122 / 151 | 0.93 |
| HONDA_CIVIC | Honda Civic | seen | 4.197 | SMALL |  | ICE | lab | 1,483 / 33 | 0 | 400 / 33 / 193 | 0.57 |
| HONDA_CIVIC_2022 | Honda Civic | seen | 4.215 | SMALL |  | ICE | lab | 1,341 / 46 | 265 | 400 / 36 / 148 | 0.73 |
| HONDA_CIVIC_BOSCH | Honda Civic | seen | 4.197 | SMALL |  | ICE | lab | 2,130 / 62 | 289 | 400 / 55 / 155 | 0.81 |
| HONDA_CRV_5G | Honda CR-V | seen | 4.277 | SMALL |  | ICE | lab | 3,836 / 100 | 280 | 400 / 92 / 160 | 0.85 |
| HONDA_ODYSSEY | Honda Odyssey | seen | 5.145 | LARGE |  | ICE | lab | 4,337 / 101 | 199 | 400 / 97 / 181 | 0.84 |
| HONDA_PILOT | Honda Pilot | seen | 4.915 | LARGE | yes | ICE | lab | 3,239 / 86 | 218 | 400 / 77 / 165 | 0.88 |
| HONDA_RIDGELINE | Honda Ridgeline | seen | 5.334 | LARGE |  | ICE | lab | 2,689 / 61 | 201 | 400 / 56 / 154 | 0.86 |
| HYUNDAI_IONIQ_5 | Hyundai Ioniq 5 | seen | 4.826 | MID |  | BEV | lab | 3,594 / 116 | 440 | 400 / 103 / 124 | 0.92 |
| HYUNDAI_PALISADE | Hyundai Palisade | seen | 4.965 | LARGE |  | ICE | lab | 2,485 / 66 | 150 | 400 / 63 / 178 | 0.75 |
| HYUNDAI_SONATA | Hyundai Sonata | seen | 4.624 | MID |  | ICE | lab | 1,465 / 40 | 100 | 400 / 38 / 153 | 0.65 |
| KIA_EV6 | Kia EV6 | seen | 4.744 | MID |  | BEV | lab | 3,062 / 102 | 529 | 400 / 88 / 139 | 0.88 |
| KIA_NIRO_EV | Kia Niro | seen | 4.267 | SMALL |  | BEV | lab | 1,662 / 42 | 157 | 400 / 38 / 183 | 0.65 |
| LEXUS_ES_TSS2 | Lexus ES | seen | 4.606 | MID |  | HEV | lab | 2,109 / 63 | 32 | 400 / 62 / 147 | 0.81 |
| MAZDA_CX5_2022 | Mazda CX-5 | seen | 4.306 | SMALL |  | ICE | lab | 3,096 / 85 | 248 | 400 / 80 / 140 | 0.89 |
| RAM_1500_5TH_GEN | RAM 1500 | seen | 6.732 | LARGE |  | ICE | lab | 4,266 / 112 | 543 | 400 / 99 / 132 | 0.95 |
| RIVIAN_R1 | Rivian R1T/R1S | seen | 5.578 | LARGE |  | BEV | lab | 5,827 / 215 | 1,535 | 400 / 97 / 97 | 1.00 |
| SUBARU_ASCENT | Subaru Ascent | seen | 4.721 | MID |  | ICE | lab | 1,459 / 34 | 0 | 400 / 34 / 178 | 0.70 |
| SUBARU_FORESTER | Subaru Forester | seen | 4.185 | SMALL |  | ICE | lab | 1,590 / 42 | 251 | 400 / 36 / 217 | 0.52 |
| SUBARU_OUTBACK | Subaru Outback | seen | 4.225 | SMALL |  | ICE | lab | 1,656 / 45 | 199 | 400 / 41 / 165 | 0.71 |
| TOYOTA_CAMRY | Toyota Camry | seen | 4.468 | MID | yes | HEV | lab | 3,430 / 86 | 0 | 400 / 86 / 166 | 0.81 |
| TOYOTA_CAMRY_TSS2 | Toyota Camry | seen | 4.468 | MID | yes | HEV | lab | 3,052 / 84 | 150 | 400 / 81 / 155 | 0.83 |
| TOYOTA_HIGHLANDER | Toyota Highlander | seen | 4.603 | MID |  | ICE | lab | 2,443 / 66 | 58 | 400 / 63 / 128 | 0.92 |
| TOYOTA_HIGHLANDER_TSS2 | Toyota Highlander | seen | 4.680 | MID |  | ICE | lab | 6,973 / 179 | 353 | 400 / 171 / 173 | 0.85 |
| TOYOTA_PRIUS | Toyota Prius | seen | 4.131 | SMALL |  | HEV | lab | 8,465 / 247 | 365 | 400 / 114 / 114 | 1.00 |
| TOYOTA_PRIUS_TSS2 | Toyota Prius | seen | 4.144 | SMALL |  | HEV | lab | 3,945 / 104 | 177 | 400 / 99 / 162 | 0.88 |
| TOYOTA_RAV4 | Toyota RAV4 | seen | 4.134 | SMALL |  | ICE | lab | 1,665 / 43 | 66 | 400 / 40 / 193 | 0.62 |
| TOYOTA_RAV4H | Toyota RAV4 | seen | 4.134 | SMALL |  | HEV | lab | 1,405 / 40 | 113 | 400 / 37 / 171 | 0.59 |
| TOYOTA_RAV4_PRIME | Toyota RAV4 | seen | 4.317 | SMALL |  | PHEV | lab | 1,471 / 45 | 171 | 400 / 41 / 143 | 0.83 |
| TOYOTA_RAV4_TSS2 | Toyota RAV4 | seen | 4.344 | SMALL |  | ICE | lab | 11,575 / 304 | 774 | 400 / 114 / 114 | 1.00 |
| TOYOTA_RAV4_TSS2_2022 | Toyota RAV4 | seen | 4.344 | SMALL |  | HEV | lab | 2,578 / 72 | 110 | 400 / 68 / 130 | 0.90 |
| TOYOTA_RAV4_TSS2_2023 | Toyota RAV4 | seen | 4.344 | SMALL |  | ICE | lab | 5,170 / 168 | 265 | 400 / 160 / 162 | 0.85 |
| VOLKSWAGEN_GOLF_MK7 | Volkswagen Golf | seen | 4.020 | SMALL |  | unknown (MIXED) | lab | 1,762 / 48 | 351 | 400 / 40 / 179 | 0.69 |
| VOLKSWAGEN_TIGUAN_MK2 | Volkswagen Tiguan | seen | 4.335 | SMALL |  | unknown (MIXED) | lab | 1,564 / 38 | 335 | 400 / 31 / 196 | 0.61 |
| CHEVROLET_VOLT | Chevrolet Volt | seen | 4.183 | SMALL |  | PHEV | lab | 1,130 / 30 | 228 | 397 / 24 / 205 | 0.40 |
| CHRYSLER_PACIFICA_2018 | Chrysler Pacifica | seen | 5.361 | LARGE |  | ICE | lab | 931 / 21 | 100 | 389 / 19 / 188 | 0.50 |
| AUDI_Q3_MK2 | Audi Q3 | seen | 4.234 | SMALL |  | ICE | lab | 434 / 12 | 50 | 384 / 11 / 195 | 0.25 |
| LEXUS_NX | Lexus NX | seen | 4.203 | SMALL |  | ICE | public | 378 / 9 | 0 | 378 / 9 / 230 | 0.15 |
| MAZDA_CX9_2021 | Mazda CX-9 | seen | 5.149 | LARGE |  | ICE | lab | 567 / 14 | 9 | 372 / 13 / 155 | 0.54 |
| HYUNDAI_TUCSON_4TH_GEN | Hyundai Tucson | seen | 4.527 | MID |  | unknown (MIXED) | lab | 1,071 / 34 | 286 | 365 / 28 / 154 | 0.58 |
| SUBARU_IMPREZA | Subaru Impreza | seen | 4.130 | SMALL |  | ICE | lab | 1,022 / 29 | 200 | 359 / 25 / 154 | 0.53 |
| SUBARU_IMPREZA_2020 | Subaru Impreza | seen | 4.130 | SMALL |  | ICE | lab | 956 / 27 | 123 | 357 / 24 / 169 | 0.51 |
| TESLA_MODEL_3 | Tesla Model 3 | seen | 4.571 | MID |  | BEV | lab | 1,025 / 39 | 371 | 357 / 29 / 130 | 0.72 |
| ACURA_RDX_3G | Acura RDX | seen | 4.502 | MID |  | ICE | lab | 747 / 21 | 52 | 344 / 18 / 196 | 0.35 |
| HONDA_INSIGHT | Honda Insight | seen | 4.201 | SMALL |  | HEV | lab | 661 / 16 | 50 | 341 / 15 / 182 | 0.37 |
| HYUNDAI_ELANTRA_HEV_2021 | Hyundai Elantra | seen | 4.311 | SMALL |  | HEV | lab | 505 / 12 | 50 | 338 / 11 / 224 | 0.09 |
| HYUNDAI_SONATA_HYBRID | Hyundai Sonata | seen | 4.624 | MID |  | HEV | lab | 591 / 15 | 100 | 336 / 13 / 176 | 0.32 |
| SKODA_SUPERB_MK3 | Skoda Superb | seen | 4.482 | MID |  | ICE | lab | 355 / 12 | 31 | 324 / 10 / 197 | 0.24 |
| LEXUS_IS | Lexus IS | seen | 4.317 | SMALL |  | ICE | lab | 323 / 8 | 0 | 323 / 8 / 204 | 0.15 |
| HYUNDAI_ELANTRA_2021 | Hyundai Elantra | seen | 4.311 | SMALL |  | ICE | lab | 454 / 15 | 135 | 319 / 12 / 150 | 0.32 |
| FORD_MUSTANG_MACH_E_MK1 | Ford Mustang Mach-E | seen | 4.845 | MID | yes | BEV | lab | 315 / 13 | 1 | 314 / 12 / 142 | 0.30 |
| HONDA_CRV_HYBRID | Honda CR-V | seen | 4.277 | SMALL |  | HEV | lab | 359 / 14 | 50 | 309 / 13 / 188 | 0.25 |
| SKODA_KODIAQ_MK1 | Skoda Kodiaq | seen | 4.411 | SMALL | yes | ICE | lab | 465 / 13 | 37 | 303 / 12 / 168 | 0.31 |
| HYUNDAI_IONIQ_6 | Hyundai Ioniq 6 | seen | 4.802 | MID |  | BEV | lab | 996 / 35 | 366 | 287 / 24 / 117 | 0.62 |
| HYUNDAI_GENESIS | Hyundai Genesis | seen | 4.896 | LARGE | yes | ICE | lab | 453 / 13 | 0 | 286 / 13 / 166 | 0.23 |
| KIA_SPORTAGE_5TH_GEN | Kia Sportage | seen | 4.421 | SMALL | yes | unknown (MIXED) | lab | 386 / 12 | 102 | 284 / 9 / 149 | 0.19 |
| HONDA_FIT | Honda Fit | seen | 3.737 | SMALL |  | ICE | lab | 282 / 7 | 0 | 282 / 7 / 118 | 0.30 |
| LEXUS_NX_TSS2 | Lexus NX | seen | 4.203 | SMALL |  | ICE | lab | 509 / 16 | 0 | 282 / 16 / 149 | 0.39 |
| FORD_ESCAPE_MK4 | Ford Escape | seen | 4.387 | SMALL | yes | unknown (MIXED) | lab | 356 / 17 | 76 | 280 / 13 / 174 | 0.16 |
| AUDI_A3_MK3 | Audi A3 | seen | 3.969 | SMALL |  | unknown (MIXED) | lab | 295 / 7 | 50 | 245 / 6 / 128 | 0.36 |
| LEXUS_IS_TSS2 | Lexus IS | seen | 4.457 | MID | yes | ICE | lab | 295 / 9 | 50 | 245 / 8 / 137 | 0.20 |
| HYUNDAI_SONATA_LF | Hyundai Sonata | seen | 4.565 | MID |  | ICE | lab | 275 / 7 | 50 | 225 / 6 / 112 | 0.35 |
| KIA_NIRO_HEV_2ND_GEN | Kia Niro | seen | 4.293 | SMALL |  | HEV | lab | 190 / 8 | 0 | 190 / 8 / 117 | 0.19 |
| TOYOTA_ALPHARD_TSS2 | Toyota Alphard | seen | 4.793 | MID |  | unknown (MIXED) | lab | 170 / 6 | 0 | 170 / 6 / 85 | 0.18 |
| LEXUS_ES | Lexus ES | seen | 4.606 | MID |  | ICE | lab | 214 / 7 | 50 | 164 / 6 / 120 | 0.02 |
| TOYOTA_CHR | Toyota C-HR | seen | 4.077 | SMALL |  | unknown (MIXED) | lab | 159 / 7 | 0 | 159 / 7 / 99 | 0.16 |
| ACURA_RDX | Acura RDX | seen | 4.315 | SMALL |  | ICE | public | 147 / 3 | 0 | 147 / 3 / 18 | 0.88 |
| KIA_NIRO_HEV_2021 | Kia Niro | seen | 4.293 | SMALL |  | HEV | lab | 146 / 4 | 0 | 146 / 4 / 98 | 0.06 |
| RAM_HD_5TH_GEN | RAM 2500/3500 HD | seen | 6.575 | LARGE |  | ICE | lab | 143 / 9 | 0 | 143 / 9 / 58 | 0.52 |
| GENESIS_G70_2020 | Genesis G70 | seen | 4.544 | MID |  | ICE | lab | 135 / 4 | 0 | 135 / 4 / 42 | 0.48 |
| VOLKSWAGEN_ARTEON_MK1 | Volkswagen Arteon | seen | 4.493 | MID |  | unknown (MIXED) | lab | 185 / 5 | 50 | 135 / 4 / 43 | 0.44 |
| KIA_NIRO_EV_2ND_GEN | Kia Niro | seen | 4.255 | SMALL |  | BEV | lab | 282 / 6 | 150 | 132 / 3 / 60 | 0.36 |
| VOLKSWAGEN_CRAFTER_MK2 | Volkswagen Crafter | seen | 6.239 | LARGE |  | unknown (MIXED) | lab | 165 / 5 | 50 | 115 / 4 / 21 | 0.53 |
| SUBARU_OUTBACK_PREGLOBAL_2018 | Subaru Outback | seen | 4.265 | SMALL |  | ICE | lab | 108 / 5 | 0 | 108 / 5 / 46 | 0.16 |
| HONDA_CRV | Honda CR-V | seen | 4.227 | SMALL |  | ICE | lab | 156 / 5 | 50 | 106 / 4 / 46 | 0.41 |
| NISSAN_ROGUE | Nissan Rogue | seen | 4.314 | SMALL |  | ICE | lab | 103 / 5 | 0 | 103 / 5 / 67 | 0.12 |
| KIA_NIRO_PHEV_2022 | Kia Niro | seen | 4.293 | SMALL |  | PHEV | lab | 163 / 4 | 63 | 100 / 2 / 76 | 0.02 |
| FORD_BRONCO_SPORT_MK1 | Ford Bronco Sport | seen | 4.279 | SMALL |  | ICE | lab | 94 / 8 | 0 | 94 / 8 / 30 | 0.60 |
| KIA_FORTE | Kia Forte | seen | 4.367 | SMALL |  | ICE | lab | 81 / 3 | 0 | 81 / 3 / 40 | 0.35 |
| VOLKSWAGEN_TRANSPORTER_T61 | Volkswagen Transporter | seen | 4.911 | LARGE | yes | ICE | lab | 75 / 2 | 0 | 75 / 2 / 50 | 0.13 |
| KIA_NIRO_PHEV | Kia Niro | seen | 4.293 | SMALL |  | PHEV | lab | 103 / 4 | 29 | 74 / 3 / 22 | 0.51 |
| LEXUS_LC_TSS2 | Lexus LC | seen | 4.738 | MID |  | ICE | lab | 121 / 3 | 50 | 71 / 2 / 54 | 0.09 |
| HYUNDAI_CUSTIN_1ST_GEN | Hyundai Custin | seen | 4.861 | MID | yes | ICE | lab | 58 / 4 | 0 | 58 / 4 / 38 | 0.09 |
| VOLKSWAGEN_TAOS_MK1 | Volkswagen Taos | seen | 4.182 | SMALL |  | ICE | lab | 55 / 2 | 0 | 55 / 2 / 24 | 0.29 |
| LEXUS_RC | Lexus RC | seen | 4.327 | SMALL |  | ICE | public | 53 / 3 | 0 | 53 / 3 / 39 | 0.07 |
| SUBARU_LEGACY | Subaru Legacy | seen | 4.239 | SMALL |  | ICE | lab | 98 / 3 | 46 | 52 / 2 / 23 | 0.12 |
| SUBARU_FORESTER_2022 | Subaru Forester | seen | 4.185 | SMALL |  | ICE | lab | 50 / 1 | 0 | 50 / 1 / 19 | 0.32 |
| CHEVROLET_TRAVERSE | Chevrolet Traverse | seen | 5.238 | LARGE |  | ICE | lab | 31 / 2 | 0 | 31 / 2 / 18 | 0.39 |
| FORD_FOCUS_MK4 | Ford Focus | seen | 4.216 | SMALL |  | ICE | lab | 124 / 3 | 100 | 24 / 1 / 9 | 0.62 |
| TOYOTA_CHR_TSS2 | Toyota C-HR | seen | 4.077 | SMALL |  | unknown (MIXED) | lab | 67 / 3 | 50 | 17 / 2 / 11 | 0.47 |
| CADILLAC_ESCALADE_ESV_2019 | Cadillac Escalade ESV | seen | 5.757 | LARGE |  | ICE | public | 8 / 1 | 0 | 8 / 1 / 2 | 0.62 |
| SKODA_KAROQ_MK1 | Skoda Karoq | seen | 4.146 | SMALL |  | ICE | lab | 58 / 4 | 50 | 8 / 3 / 8 | 0.00 |
| CHEVROLET_VOLT_2019 | Chevrolet Volt | seen | 4.189 | SMALL |  | PHEV | lab | 55 / 2 | 50 | 5 / 1 / 1 | 0.40 |
| ACURA_ILX | Acura ILX | seen | 4.056 | SMALL |  | ICE | lab | 4 / 1 | 0 | 4 / 1 / 3 | 0.00 |
| MAZDA_CX5 | Mazda CX-5 | seen | 4.306 | SMALL |  | ICE | lab | 4 / 1 | 0 | 4 / 1 / 3 | 0.00 |
| HONDA_CRV_HYBRID_6G | Honda CR-V | seen | 4.371 | SMALL |  | HEV | lab | 3 / 1 | 0 | 3 / 1 / 3 | 0.00 |
| HONDA_E | Honda e | seen | 3.795 | SMALL |  | BEV | public | 2 / 1 | 0 | 2 / 1 / 1 | 1.00 |
| CADILLAC_XT4 | Cadillac XT4 | seen | 4.458 | MID | yes | ICE | lab | 6 / 2 | 5 | 1 / 1 / 1 | 0.00 |
| FORD_ESCAPE_MK4_5 | Ford Escape | seen | 4.387 | SMALL | yes | unknown (MIXED) | lab | 1 / 1 | 0 | 1 / 1 / 1 | 0.00 |
| HYUNDAI_SONATA_HEV_2024 | Hyundai Sonata | seen | 4.624 | MID |  | HEV | lab | 1 / 1 | 0 | 1 / 1 / 1 | 0.00 |
| GENESIS_GV70_ELECTRIFIED_1ST_GEN | Genesis GV70 Electrified | seen | 4.744 | MID |  | BEV | lab | 50 / 1 | 50 | 0 / 0 / 0 | - |
| HONDA_PILOT_4G | Honda Pilot | seen | 4.968 | LARGE |  | ICE | lab | 2 / 1 | 2 | 0 / 0 / 0 | - |
| KIA_SELTOS | Kia Seltos | seen | 4.155 | SMALL |  | ICE | lab | 17 / 1 | 17 | 0 / 0 / 0 | - |
| NISSAN_XTRAIL | Nissan X-Trail | seen | 4.247 | SMALL |  | ICE | lab | 50 / 1 | 50 | 0 / 0 / 0 | - |
| SUBARU_OUTBACK_PREGLOBAL | Subaru Outback | seen | 4.265 | SMALL |  | ICE | lab | 50 / 1 | 50 | 0 / 0 / 0 | - |
| TOYOTA_COROLLA_TSS2 | Toyota Corolla | held-out | 4.093 | SMALL |  | HEV | lab | 8,612 / 255 | 929 | 7,683 / 234 / 3967 | 0.25 |
| LEXUS_RX_TSS2 | Lexus RX | held-out | 4.550 | MID |  | ICE | lab | 2,589 / 71 | 24 | 2,565 / 69 / 1457 | 0.21 |
| TOYOTA_SIENNA | Toyota Sienna | held-out | 5.212 | LARGE |  | ICE | lab | 2,331 / 66 | 199 | 2,132 / 62 / 1208 | 0.19 |
| LEXUS_RX | Lexus RX | held-out | 4.550 | MID |  | HEV | lab | 2,084 / 58 | 113 | 1,971 / 55 / 1023 | 0.24 |
| VOLKSWAGEN_ATLAS_MK1 | Volkswagen Atlas | held-out | 5.108 | LARGE |  | ICE | lab | 1,509 / 34 | 100 | 1,409 / 32 / 785 | 0.23 |
| TOYOTA_COROLLA | Toyota Corolla | held-out | 4.139 | SMALL |  | ICE | lab | 1,296 / 39 | 0 | 1,296 / 39 / 701 | 0.28 |
| HYUNDAI_SANTA_FE_2022 | Hyundai Santa Fe | held-out | 4.567 | MID |  | ICE | lab | 1,018 / 24 | 63 | 955 / 22 / 476 | 0.27 |
| TOYOTA_SIENNA_4TH_GEN | Toyota Sienna | held-out | 5.296 | LARGE |  | HEV | lab | 923 / 36 | 25 | 898 / 33 / 417 | 0.30 |
| HYUNDAI_SANTA_FE | Hyundai Santa Fe | held-out | 4.531 | MID |  | ICE | lab | 979 / 26 | 100 | 879 / 24 / 487 | 0.24 |
| NISSAN_LEAF | Nissan Leaf | held-out | 4.160 | SMALL |  | BEV | lab | 1,172 / 31 | 346 | 826 / 23 / 489 | 0.14 |
| JEEP_GRAND_CHEROKEE | Jeep Grand Cherokee | held-out | 4.405 | SMALL | yes | ICE | lab | 869 / 23 | 50 | 819 / 22 / 442 | 0.24 |
| TOYOTA_AVALON_2019 | Toyota Avalon | held-out | 4.526 | MID |  | HEV | lab | 830 / 26 | 33 | 797 / 25 / 468 | 0.18 |
| JEEP_GRAND_CHEROKEE_2019 | Jeep Grand Cherokee | held-out | 4.405 | SMALL | yes | ICE | lab | 652 / 16 | 1 | 651 / 15 / 376 | 0.17 |
| TESLA_MODEL_Y | Tesla Model Y | held-out | 4.728 | MID |  | BEV | lab | 826 / 30 | 282 | 544 / 21 / 272 | 0.31 |
| KIA_CARNIVAL_4TH_GEN | Kia Carnival | held-out | 5.302 | LARGE |  | ICE | lab | 703 / 20 | 200 | 503 / 15 / 303 | 0.14 |
| HYUNDAI_KONA_EV | Hyundai Kona | held-out | 4.025 | SMALL |  | BEV | lab | 505 / 16 | 100 | 405 / 14 / 168 | 0.35 |
| KIA_STINGER | Kia Stinger | held-out | 4.508 | MID |  | ICE | lab | 434 / 9 | 50 | 384 / 8 / 161 | 0.26 |
| VOLKSWAGEN_JETTA_MK7 | Volkswagen Jetta | held-out | 4.168 | SMALL |  | ICE | lab | 399 / 13 | 50 | 349 / 12 / 186 | 0.33 |
| HYUNDAI_SANTA_FE_PHEV_2022 | Hyundai Santa Fe | held-out | 4.567 | MID |  | PHEV | lab | 313 / 8 | 0 | 313 / 8 / 139 | 0.23 |
| HYUNDAI_IONIQ_HEV_2022 | Hyundai Ioniq | held-out | 4.239 | SMALL |  | HEV | lab | 312 / 8 | 0 | 312 / 8 / 159 | 0.28 |
| GENESIS_G80 | Genesis G80 | held-out | 4.947 | LARGE |  | ICE | public | 311 / 7 | 0 | 311 / 7 / 151 | 0.26 |
| KIA_SORENTO_HEV_4TH_GEN | Kia Sorento | held-out | 4.639 | MID |  | HEV | lab | 308 / 8 | 0 | 308 / 8 / 175 | 0.19 |
| HYUNDAI_IONIQ_PHEV | Hyundai Ioniq | held-out | 4.217 | SMALL |  | PHEV | lab | 309 / 10 | 10 | 299 / 9 / 165 | 0.24 |
| HYUNDAI_SANTA_FE_HEV_2022 | Hyundai Santa Fe | held-out | 4.567 | MID |  | HEV | lab | 318 / 9 | 50 | 268 / 8 / 134 | 0.29 |
| VOLKSWAGEN_PASSAT_MK8 | Volkswagen Passat | held-out | 4.397 | SMALL | yes | unknown (MIXED) | lab | 413 / 12 | 151 | 262 / 8 / 127 | 0.26 |
| GENESIS_G90 | Genesis G90 | held-out | 5.180 | LARGE |  | ICE | lab | 260 / 10 | 4 | 256 / 9 / 127 | 0.24 |
| KIA_SORENTO_4TH_GEN | Kia Sorento | held-out | 4.639 | MID |  | ICE | lab | 306 / 9 | 50 | 256 / 8 / 161 | 0.23 |
| TOYOTA_AVALON_TSS2 | Toyota Avalon | held-out | 4.526 | MID |  | unknown (UNK) | public | 254 / 6 | 0 | 254 / 6 / 168 | 0.15 |
| TOYOTA_AVALON | Toyota Avalon | held-out | 4.470 | MID | yes | ICE | lab | 252 / 7 | 0 | 252 / 7 / 130 | 0.25 |
| CHEVROLET_EQUINOX | Chevrolet Equinox | held-out | 4.299 | SMALL |  | ICE | lab | 303 / 7 | 53 | 250 / 5 / 166 | 0.10 |
| HYUNDAI_IONIQ_EV_2020 | Hyundai Ioniq | held-out | 4.211 | SMALL |  | BEV | lab | 288 / 6 | 50 | 238 / 5 / 120 | 0.25 |
| GENESIS_GV70_1ST_GEN | Genesis GV70 | held-out | 4.733 | MID |  | ICE | lab | 451 / 13 | 250 | 201 / 8 / 102 | 0.34 |
| NISSAN_ALTIMA | Nissan Altima | held-out | 4.561 | MID |  | ICE | lab | 200 / 4 | 0 | 200 / 4 / 105 | 0.22 |
| KIA_OPTIMA_G4_FL | Kia Optima | held-out | 4.529 | MID |  | ICE | lab | 167 / 5 | 0 | 167 / 5 / 73 | 0.38 |
| HYUNDAI_KONA_EV_2022 | Hyundai Kona | held-out | 4.025 | SMALL |  | BEV | lab | 206 / 5 | 50 | 156 / 4 / 92 | 0.28 |
| GENESIS_GV80 | Genesis GV80 | held-out | 4.960 | LARGE |  | ICE | lab | 200 / 4 | 50 | 150 / 3 / 95 | 0.17 |
| FORD_F_150_LIGHTNING_MK1 | Ford F-150 Lightning | held-out | 6.410 | LARGE |  | BEV | lab | 230 / 9 | 105 | 125 / 6 / 34 | 0.43 |
| NISSAN_LEAF_IC | Nissan Leaf | held-out | 4.160 | SMALL |  | BEV | lab | 166 / 6 | 50 | 116 / 5 / 88 | 0.08 |
| HYUNDAI_IONIQ | Hyundai Ioniq | held-out | 4.239 | SMALL |  | HEV | lab | 159 / 6 | 53 | 106 / 4 / 47 | 0.24 |
| KIA_K5_2021 | Kia K5 | held-out | 4.600 | MID |  | ICE | lab | 101 / 3 | 0 | 101 / 3 / 69 | 0.11 |
| GENESIS_GV60_EV_1ST_GEN | Genesis GV60 | held-out | 4.734 | MID |  | BEV | lab | 199 / 4 | 99 | 100 / 2 / 43 | 0.16 |
| HONDA_HRV | Honda HR-V | held-out | 4.010 | SMALL |  | ICE | lab | 150 / 3 | 50 | 100 / 2 / 62 | 0.13 |
| KIA_STINGER_2022 | Kia Stinger | held-out | 4.508 | MID |  | ICE | public | 100 / 2 | 0 | 100 / 2 / 27 | 0.37 |
| FORD_RANGER_MK2 | Ford Ranger | held-out | 5.301 | LARGE |  | ICE | lab | 124 / 4 | 42 | 82 / 3 / 36 | 0.45 |
| HONDA_HRV_3G | Honda HR-V | held-out | 4.167 | SMALL |  | ICE | lab | 60 / 3 | 0 | 60 / 3 / 17 | 0.27 |
| HYUNDAI_SANTA_CRUZ_1ST_GEN | Hyundai Santa Cruz | held-out | 4.936 | LARGE |  | ICE | lab | 106 / 3 | 56 | 50 / 1 / 46 | 0.00 |
| KIA_SORENTO | Kia Sorento | held-out | 4.540 | MID |  | ICE | public | 10 / 1 | 0 | 10 / 1 / 1 | 1.00 |
| HYUNDAI_AZERA_HEV_6TH_GEN | Hyundai Azera | held-out | 4.633 | MID |  | HEV | lab | 5 / 1 | 0 | 5 / 1 / 5 | 0.00 |
| HYUNDAI_KONA_2022 | Hyundai Kona | held-out | 4.025 | SMALL |  | ICE | lab | 24 / 3 | 22 | 2 / 2 / 2 | 0.00 |
| GMC_YUKON | GMC Yukon | held-out | 5.139 | LARGE |  | ICE | lab | 24 / 1 | 24 | 0 / 0 / 0 | - |
| HYUNDAI_KONA_EV_2ND_GEN | Hyundai Kona | held-out | 4.176 | SMALL |  | BEV | lab | 50 / 1 | 50 | 0 / 0 / 0 | - |
| KIA_CEED | Kia Ceed | held-out | 4.150 | SMALL |  | ICE | lab | 50 / 1 | 50 | 0 / 0 / 0 | - |
| KIA_OPTIMA_H_G4_FL | Kia Optima | held-out | 4.529 | MID |  | HEV | lab | 50 / 1 | 50 | 0 / 0 / 0 | - |
| TOYOTA_YARIS | Toyota Yaris | held-out | 3.782 | SMALL |  | HEV | lab | 50 / 1 | 50 | 0 / 0 / 0 | - |
| VOLKSWAGEN_PASSAT_NMS | Volkswagen Passat | held-out | 4.416 | SMALL | yes | ICE | lab | 33 / 3 | 33 | 0 / 0 / 0 | - |
| SKODA_OCTAVIA_MK3 | Skoda Octavia | absent | 4.149 | SMALL |  | ICE | public | 322 / 7 | 0 | 322 / 7 / 133 | 0.46 |
| TOYOTA_MIRAI | Toyota Mirai | absent | 4.281 | SMALL |  | FCEV | public | 299 / 6 | 0 | 299 / 6 / 140 | 0.28 |
| SEAT_ATECA_MK1 | SEAT Ateca | absent | 4.111 | SMALL |  | ICE | public | 153 / 4 | 0 | 153 / 4 / 79 | 0.16 |
| VOLKSWAGEN_TROC_MK1 | Volkswagen T-Roc | absent | 3.987 | SMALL |  | ICE | public | 150 / 3 | 0 | 150 / 3 / 82 | 0.12 |
| HYUNDAI_ELANTRA_GT_I30 | Hyundai Elantra GT / i30 | absent | 4.126 | SMALL |  | ICE | public | 100 / 2 | 0 | 100 / 2 / 51 | 0.19 |
| HYUNDAI_IONIQ_PHEV_2019 | Hyundai Ioniq Plug-in | absent | 4.217 | SMALL |  | PHEV | public | 100 / 2 | 0 | 100 / 2 / 35 | 0.14 |
| LEXUS_CTH | Lexus CT 200h | absent | 3.985 | SMALL |  | HEV | public | 65 / 3 | 0 | 65 / 3 / 42 | 0.09 |
| TOYOTA_PRIUS_V | Toyota Prius v | absent | 4.288 | SMALL |  | HEV | public | 53 / 3 | 0 | 53 / 3 / 17 | 0.24 |
| BUICK_REGAL | Buick Regal Sportback | absent | 4.521 | MID |  | ICE | public | 50 / 1 | 0 | 50 / 1 / 33 | 0.00 |
| HONDA_FREED | Honda Freed | absent | 4.062 | SMALL |  | unknown (UNK) | public | 50 / 1 | 0 | 50 / 1 / 32 | 0.00 |
| HYUNDAI_STARIA_4TH_GEN | Hyundai Staria | absent | 5.651 | LARGE |  | ICE | public | 61 / 2 | 11 | 50 / 1 / 15 | 0.54 |
| VOLKSWAGEN_TOURAN_MK2 | Volkswagen Touran | absent | 4.389 | SMALL | yes | ICE | lab | 50 / 1 | 0 | 50 / 1 / 28 | 0.38 |
| SKODA_KAMIQ_MK1 | Skoda Kamiq | absent | 4.039 | SMALL |  | ICE | public | 35 / 1 | 0 | 35 / 1 / 9 | 0.66 |
| DODGE_DURANGO | Dodge Durango | absent | 4.946 | LARGE |  | ICE | public | 15 / 1 | 0 | 15 / 1 / 1 | 0.87 |
| CADILLAC_ESCALADE | Cadillac Escalade | absent | 5.136 | LARGE |  | ICE | public | 1 / 1 | 0 | 1 / 1 / 1 | 0.00 |
| GMC_ACADIA | GMC Acadia | absent | 4.680 | MID |  | ICE | public | 50 / 1 | 50 | 0 / 0 / 0 | - |
| VOLKSWAGEN_POLO_MK6 | Volkswagen Polo | absent | 3.863 | SMALL |  | ICE | lab | 13 / 1 | 13 | 0 / 0 / 0 | - |

## Budget alternative

`select_budget.txt` (same folder as `select.txt`) keeps every 'seen' and 'absent' selection but caps the held-out platforms at 1000 segments each with the same >= 20-device round-robin rule: 49,563 segments from 4,334 devices and 23,302 routes, about 78.3 GB (held-out stratum 19,420 segments from 880 devices). Use it when the full list (95.8 GB) is too much for one sitting; the two lists are nested (the budget list is a subset), so a budget download can be topped up later with `select.txt` and the same command.

Per-platform budget counts (held-out platforms whose selection differs):

| platform | model | selected | budget |
|---|---|---|---|
| TOYOTA_COROLLA_TSS2 | Toyota Corolla | 7,683 | 1,000 |
| LEXUS_RX_TSS2 | Lexus RX | 2,565 | 1,000 |
| TOYOTA_SIENNA | Toyota Sienna | 2,132 | 1,000 |
| LEXUS_RX | Lexus RX | 1,971 | 1,000 |
| VOLKSWAGEN_ATLAS_MK1 | Volkswagen Atlas | 1,409 | 1,000 |
| TOYOTA_COROLLA | Toyota Corolla | 1,296 | 1,000 |

## Dropped index keys

| key | folded | reason | segments |
|---|---|---|---|
| MOCK | MOCK | mock | 107 |
| PERODUA MYVI | PERODUA MYVI | fork / unknown to opendbc and to the label tables | 12 |
| TURBO_RC_CAR | TURBO_RC_CAR | fork / unknown to opendbc and to the label tables | 3 |

Platforms with no segment left after the shared-device removal: GENESIS_GV70_ELECTRIFIED_1ST_GEN, HONDA_PILOT_4G, KIA_SELTOS, NISSAN_XTRAIL, SUBARU_OUTBACK_PREGLOBAL, GMC_YUKON, HYUNDAI_KONA_EV_2ND_GEN, KIA_CEED, KIA_OPTIMA_H_G4_FL, TOYOTA_YARIS, VOLKSWAGEN_PASSAT_NMS, GMC_ACADIA, VOLKSWAGEN_POLO_MK6.

## Download (owner's machine)

Resumable, 8 parallel streams, 5 retries per file; `$EXT` is the external-data folder on the large data disk (`select.txt` was written there by `ccs_select.py`). `download_owner_sets.sh --only ccs` runs the same loop and keeps a failure list in `$EXT/logs/ccs_failed.txt`.

```bash
export EXT=<folder for external data on the large data disk>
curl -sfL --create-dirs -o $EXT/ccs/database.json https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main/database.json   # index, 9.4 MB (already present when ccs_select.py ran)
# $EXT/ccs/select.txt: built by  $CPY experiments/transfer/ccs_select.py --index $EXT/ccs/database.json --out $EXT/ccs/select.txt
xargs -P 8 -I{} sh -c 'curl -sfL -C - --retry 5 --retry-delay 5 --create-dirs -o "$EXT/ccs/segments/{}/rlog.zst" "https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main/segments/{}/rlog.zst"' < $EXT/ccs/select.txt
# equivalent: hf download commaai/commaCarSegments --repo-type dataset --local-dir $EXT/ccs --include 'segments/<dongle>/<route>/<seg>/rlog.zst' ... (one pattern per line of select.txt)
# afterwards: find $EXT/ccs/segments -name rlog.zst -size +0 | wc -l   # should equal the line count of select.txt
```

Budget: about 1.58 MB per segment -> 95.8 GB for the whole list (the full dataset, 299 GB, is not needed). Decoding: `experiments/transfer/ccs_decode.py` (see `CCS_PIPELINE.md`).
