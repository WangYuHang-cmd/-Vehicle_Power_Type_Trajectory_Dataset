# Size classes (locked thresholds)

locked on 6 Nov 2026, before any classifier was trained on the data.

## Definition

footprint = wheelbase [m] x (front track + rear track) / 2 [m], per vehicle platform (generation) as recorded in
`vtt/specs_data/platform_specs.csv`. A route's `size_class` is cut from its platform's footprint:

| class | footprint range (m^2) |
|---|---|
| SMALL | footprint < 4.43 |
| MID | 4.43 <= footprint < 4.88 |
| LARGE | footprint >= 4.88 |

`config.FOOTPRINT_THRESHOLDS_M2 = (4.43, 4.88)`; `config.SIZE_CLASSES_SHA256` is the sha256 of this file and
`scripts/s08_splits.py` refuses to run when it differs. Curb mass and body style are published as separate columns
and take no part in the class.

## Rationale

Rule used to choose the two thresholds (fixed from the label inventory alone, before any model was trained):

1. **Candidate bands.** Lower threshold in [4.25, 4.50] m^2, upper threshold in [4.60, 5.00] m^2, on a 0.005 m^2 grid.
   The bands span the hours-tertile and natural-break candidates of `build/SIZE_INVENTORY.md` (4.30 / 4.45 and
   4.62 / 4.67) and the round proposal 4.35 / 4.85.
2. **Support constraint.** Every size class must hold, for BEV and for ICE separately, at least 100 h, 3 car models and
   5 identity groups with a drive table, so that the size task can be evaluated inside each powertrain and the
   powertrain task inside each size class. The unconstrained schemes fail this: hours tertiles leave MID x BEV with
   17 h and the natural breaks with 1 h, because the BEV fleet is either compact (Bolt, Niro, Born) or large (Rivian).
3. **Balance.** Among the admissible pairs, take those that maximise the smallest of the six size x {BEV, ICE} cells
   (309 h). This plateau is lower in [4.385, 4.50] and upper in [4.805, 5.00].
4. **Boundary placement inside the plateau.** The lower threshold sits at the midpoint of the widest gap between
   platform footprints (Kia Sportage 5th gen 4.421 | Acura TLX 1st gen 4.439), so no platform is within 0.008 m^2 of
   it. The upper threshold sits inside the gap that separates the two-row mid-size vehicles (Ioniq 5 facelift 4.875
   is the largest) from the first full-size sedan, three-row SUV and pickup (Hyundai Genesis 4.896, Honda Pilot 4.915,
   Ford Maverick 4.929), rounded to 0.01 m^2; it keeps both Ioniq 5 generations in one class. The widest gap of the upper
   plateau (4.973 to 5.06) was not used because it would put the three-row Palisade, Pilot, MDX 4th gen, GV80 and the
   Tesla Model S into MID.
5. **Reading.** SMALL = compact cars and compact crossovers (Civic, Corolla, Prius, Elantra, RAV4, CR-V, Tiguan,
   Escape, Sportage, Bolt, Niro, Born); MID = mid-size sedans and two-row mid-size SUVs (Camry, Accord, Sonata, Tucson,
   Santa Fe, RDX, MDX 3rd gen, Model 3, Model Y, EV6, GV60, Ioniq 5, Ioniq 6, Mach-E); LARGE = full-size sedans,
   three-row SUVs, minivans and pickups (Palisade, Pilot, Explorer, MDX 4th gen, Model S, Model X, Rivian R1, EV9,
   Sienna, F-150, RAM).

## Cell table (routes with a drive table)

| size \ powertrain | ICE | HEV | PHEV | BEV | null | total |
|---|---|---|---|---|---|---|
| SMALL | 749 h / 55 m / 265 g | 319 h / 17 m / 115 g | 189 h / 9 m / 49 g | 309 h / 15 m / 99 g | 218 h / 13 m / 92 g | 1784 h / 4146 routes |
| MID | 371 h / 42 m / 136 g | 149 h / 14 m / 59 g | 0 h / 2 m / 3 g | 494 h / 10 m / 146 g | 110 h / 9 m / 38 g | 1124 h / 2592 routes |
| LARGE | 457 h / 28 m / 151 g | 7 h / 3 m / 12 g | 10 h / 1 m / 9 g | 460 h / 8 m / 67 g | 102 h / 4 m / 31 g | 1037 h / 2328 routes |
| null | 4 h / 8 m / 8 g | - | 0 h / 1 m / 1 g | 1 h / 2 m / 2 g | 46 h / 2 m / 98 g | 51 h / 191 routes |

cells: hours h / distinct car models m / identity groups g (devices after merging); null = label unknown

| class | hours | share |
|---|---|---|
| SMALL | 1784 | 44.7 % |
| MID | 1124 | 28.1 % |
| LARGE | 1037 | 25.9 % |
| null | 51 | 1.3 % |

## Platforms within +-0.05 m^2 of 4.43

| platform | model | powertrains | footprint | hours | class |
|---|---|---|---|---|---|
| POLESTAR_2 | Polestar 2 | BEV | 4.380 | 5.4 | SMALL |
| SUBARU_OUTBACK_2026 | Subaru Outback | ICE | 4.385 | 2.2 | SMALL |
| HYUNDAI_TUCSON | Hyundai Tucson | ICE | 4.385 | 0.3 | SMALL |
| FORD_ESCAPE_MK4 | Ford Escape | - | 4.387 | 9.5 | SMALL |
| FORD_ESCAPE_MK5 | Ford Escape | - | 4.387 | 3.0 | SMALL |
| FORD_ESCAPE_MK4_5 | Ford Escape | - | 4.387 | 66.7 | SMALL |
| VOLKSWAGEN_TOURAN_MK2 | Volkswagen Touran | ICE | 4.389 | 0.4 | SMALL |
| VOLKSWAGEN_PASSAT_MK8 | Volkswagen Passat | - | 4.397 | 16.2 | SMALL |
| JEEP_GRAND_CHEROKEE | Jeep Grand Cherokee | ICE | 4.405 | 2.2 | SMALL |
| JEEP_GRAND_CHEROKEE_2019 | Jeep Grand Cherokee | ICE | 4.405 | 1.6 | SMALL |
| HONDA_ACCORD_HYBRID_2016-17_SERIAL_STEERING | Honda Accord | HEV | 4.407 | 0.3 | SMALL |
| SKODA_KODIAQ_MK1 | Skoda Kodiaq | ICE | 4.411 | 7.0 | SMALL |
| VOLKSWAGEN_PASSAT_NMS | Volkswagen Passat | ICE | 4.416 | 23.9 | SMALL |
| KIA_SPORTAGE_HEV_2026 | Kia Sportage | HEV | 4.419 | 23.6 | SMALL |
| KIA_SPORTAGE_PLUG-IN_HYBRID_5TH_GEN | Kia Sportage | PHEV | 4.419 | 1.5 | SMALL |
| KIA_SPORTAGE_2026 | Kia Sportage | ICE | 4.419 | 0.9 | SMALL |
| KIA_SPORTAGE_5TH_GEN | Kia Sportage | - | 4.421 | 7.0 | SMALL |
| ACURA_TLX | Acura TLX | ICE | 4.439 | 1.8 | MID |
| ACURA_TLX_1G | Acura TLX | ICE | 4.439 | 14.5 | MID |
| LEXUS_IS_TSS2 | Lexus IS | ICE | 4.457 | 2.1 | MID |
| CADILLAC_XT4 | Cadillac XT4 | ICE | 4.458 | 1.4 | MID |
| TOYOTA_CAMRY_TSS3 | Toyota Camry | HEV | 4.461 | 1.3 | MID |
| PSA_PEUGEOT_508 | Peugeot 508 | ICE | 4.466 | 1.5 | MID |
| TOYOTA_CAMRY_TSS2 | Toyota Camry | HEV/ICE | 4.468 | 28.4 | MID |
| TOYOTA_CAMRY | Toyota Camry | HEV/ICE | 4.468 | 0.7 | MID |
| TOYOTA_AVALON | Toyota Avalon | ICE | 4.470 | 0.9 | MID |
| MAZDA_6 | Mazda 6 | ICE | 4.471 | 0.1 | MID |
| GWM_HAVAL_H6_MK4 | GWM Haval H6 | ICE | 4.478 | 4.8 | MID |
| GWM_HAVAL_H6 | GWM Haval H6 | ICE | 4.478 | 2.0 | MID |

## Platforms within +-0.05 m^2 of 4.88

| platform | model | powertrains | footprint | hours | class |
|---|---|---|---|---|---|
| FORD_MUSTANG_MACH_E_MK1 | Ford Mustang Mach-E | BEV | 4.845 | 48.6 | MID |
| HYUNDAI_CUSTIN_1ST_GEN | Hyundai Custin | ICE | 4.861 | 0.8 | MID |
| FORD_BRONCO_MK6 | Ford Bronco | ICE | 4.864 | 0.2 | MID |
| HYUNDAI_IONIQ_5_PE | Hyundai Ioniq 5 | BEV | 4.875 | 22.4 | MID |
| HYUNDAI_GENESIS | Hyundai Genesis | ICE | 4.896 | 2.7 | LARGE |
| VOLKSWAGEN_TRANSPORTER_T61 | Volkswagen Transporter | ICE | 4.911 | 0.9 | LARGE |
| HONDA_PILOT | Honda Pilot | ICE | 4.915 | 2.6 | LARGE |
| FORD_MAVERICK_MK1 | Ford Maverick | - | 4.929 | 17.7 | LARGE |

## Models whose generations fall in two classes

| model | class | platforms | hours |
|---|---|---|---|
| Acura MDX | LARGE | ACURA_MDX_4G, ACURA_MDX_4G_MMR, ACURA_MDX_4G_TYPE_S | 38.8 |
| Acura MDX | MID | ACURA_MDX_3G, ACURA_MDX_3G_HYBRID, ACURA_MDX_3G_MMR | 122.7 |
| Honda Accord | MID | HONDA_ACCORD, HONDA_ACCORD_11G | 67.5 |
| Honda Accord | SMALL | HONDA_ACCORD_HYBRID_2016-17_SERIAL_STEERING | 0.3 |
| Honda Odyssey | LARGE | HONDA_ODYSSEY, HONDA_ODYSSEY_2021, HONDA_ODYSSEY_5G_MMR | 67.1 |
| Honda Odyssey | MID | HONDA_ODYSSEY/ELYSION_CUSTOM, HONDA_ODYSSEY_CHN, HONDA_ODYSSEY_RC5, HONDA_ODYSSEY_SINGAPORE, HONDA_ODYSSEY_TWN | 9.9 |
| Hyundai Ioniq 5 | LARGE | HYUNDAI_IONIQ_5_N | 1.9 |
| Hyundai Ioniq 5 | MID | HYUNDAI_IONIQ_5, HYUNDAI_IONIQ_5_PE | 109.6 |
| Hyundai Tucson | MID | HYUNDAI_TUCSON_2025, HYUNDAI_TUCSON_4TH_GEN, HYUNDAI_TUCSON_HEV_2025 | 74.6 |
| Hyundai Tucson | SMALL | HYUNDAI_TUCSON | 0.3 |
| Jeep Grand Cherokee | MID | JEEP_GRAND_CHEROKEE_SRT_2016 | 0.5 |
| Jeep Grand Cherokee | SMALL | JEEP_GRAND_CHEROKEE, JEEP_GRAND_CHEROKEE_2019 | 3.9 |
| Lexus IS | MID | LEXUS_IS_TSS2 | 2.1 |
| Lexus IS | SMALL | LEXUS_IS | 3.8 |

## Models per class (hours with a drive table)

### SMALL (1784 h, 79 models)

Honda N-Box (3.20, 1 h), Hyundai Inster (3.66, 1 h), Honda Fit (3.73, 9 h), Toyota Yaris (3.78, 2 h), Honda e (3.79, 0 h), Hyundai Venue (3.81, 0 h), Opel Corsa (3.81, 5 h), Peugeot 208 (3.81, 6 h), Honda City (3.86, 6 h), Volkswagen Polo (3.86, 1 h), Chevrolet Bolt EV (3.90, 67 h), Subaru BRZ (3.95, 2 h), Jeep Renegade (3.96, 1 h), BMW 1 Series (E82) (3.96, 1 h), Audi A3 (3.97, 2 h), Chevrolet Bolt EUV (3.99, 46 h), Honda HR-V (4.01, 8 h), Volkswagen Caddy (4.01, 1 h), Volkswagen Golf (4.02, 45 h), Hyundai Kona (4.02, 53 h), Acura ILX (4.06, 1 h), Toyota C-HR (4.08, 16 h), Chevrolet Trailblazer (4.10, 2 h), Volkswagen Jetta (4.12, 9 h), Subaru Impreza (4.13, 23 h), Toyota Corolla (4.14, 85 h), Toyota Prius (4.14, 113 h), Skoda Karoq (4.15, 1 h), Subaru Crosstrek (4.15, 16 h), Kia Ceed (4.15, 17 h), Kia Seltos (4.16, 2 h), Nissan Leaf (4.16, 23 h), Kia XCeed (4.16, 0 h), Volkswagen Taos (4.18, 7 h), Subaru Forester (4.19, 28 h), Chevrolet Volt (4.19, 96 h), Honda Civic (4.20, 295 h), Honda Insight (4.20, 8 h), Lexus NX (4.20, 5 h), Honda Prelude (4.21, 0 h), Ford Focus (4.22, 2 h), Hyundai Ioniq (4.22, 15 h), Cupra Born (4.22, 32 h), Audi Q3 (4.23, 1 h), Acura ADX (4.24, 4 h), Acura Integra (4.24, 11 h), Nissan X-Trail (4.25, 0 h), Subaru Outback (4.27, 20 h), Ford Bronco Sport (4.28, 4 h), BYD Atto 3 (4.29, 2 h), Kia Niro (4.29, 152 h), Jeep Cherokee (4.30, 5 h), Honda CR-V (4.30, 42 h), Chevrolet Equinox (4.30, 3 h), Subaru Legacy (4.30, 24 h), Mazda CX-5 (4.31, 10 h), Volkswagen ID.3 (4.31, 1 h), Hyundai Elantra (4.31, 109 h), Nissan Rogue (4.31, 1 h), Toyota RAV4 (4.32, 129 h), Lexus IS (4.32, 4 h), Lexus RC (4.33, 0 h), Kia K4 (4.33, 0 h), Volkswagen Tiguan (4.33, 18 h), Citroen Berlingo (4.34, 0 h), Honda Clarity (4.36, 10 h), Skoda Enyaq (4.36, 6 h), Volkswagen ID.4 (4.36, 4 h), Kia Forte (4.37, 3 h), Audi A4 (4.37, 1 h), Polestar 2 (4.38, 5 h), Hyundai Tucson (4.39, 0 h), Ford Escape (4.39, 79 h), Volkswagen Touran (4.39, 0 h), Jeep Grand Cherokee (4.41, 4 h), Volkswagen Passat (4.41, 40 h), Honda Accord (4.41, 0 h), Skoda Kodiaq (4.41, 7 h), Kia Sportage (4.42, 33 h)

### MID (1124 h, 56 models)

Lexus IS (4.46, 2 h), Cadillac XT4 (4.46, 1 h), Peugeot 508 (4.47, 2 h), Toyota Camry (4.47, 30 h), Mazda 6 (4.47, 0 h), GWM Haval H6 (4.48, 7 h), Skoda Superb (4.48, 15 h), Volkswagen Arteon (4.49, 1 h), Chevrolet Malibu (4.50, 1 h), Toyota Avalon (4.50, 2 h), Acura RDX (4.50, 27 h), Kia Stinger (4.51, 15 h), Hyundai Nexo (4.52, 0 h), Kia Optima (4.53, 2 h), Audi Q5 (4.53, 2 h), Genesis G70 (4.54, 27 h), Honda Odyssey (4.54, 10 h), MG Marvel R (4.54, 1 h), Honda Accord (4.55, 67 h), Lexus RX (4.55, 11 h), Nissan Altima (4.56, 4 h), Acura TLX (4.56, 20 h), Hyundai Santa Fe (4.57, 12 h), Hyundai Azera (4.57, 2 h), Tesla Model 3 (4.57, 91 h), Hyundai Tucson (4.57, 75 h), Kia K7 (4.59, 0 h), Kia K5 (4.61, 2 h), Lexus ES (4.61, 8 h), Alfa Romeo Stelvio (4.62, 1 h), Hyundai Sonata (4.62, 74 h), Volvo V60 (4.62, 6 h), Kia Sorento (4.64, 7 h), Porsche Macan (4.64, 9 h), Toyota Highlander (4.64, 16 h), Volkswagen Sharan (4.65, 7 h), Buick LaCrosse (4.66, 2 h), Kia K8 (4.70, 2 h), Subaru Ascent (4.72, 15 h), Ford Edge (4.73, 1 h), Tesla Model Y (4.73, 22 h), Genesis GV70 (4.73, 15 h), Genesis GV60 (4.73, 27 h), Lexus LC (4.74, 0 h), Genesis GV70 Electrified (4.74, 1 h), Kia EV6 (4.74, 74 h), Acura MDX (4.75, 123 h), Luxgen n7 (4.76, 1 h), Jeep Grand Cherokee (4.78, 1 h), Toyota Alphard (4.79, 3 h), Hyundai Ioniq 6 (4.80, 119 h), Cadillac XT6 (4.82, 3 h), Ford Mustang Mach-E (4.84, 49 h), Hyundai Ioniq 5 (4.85, 110 h), Hyundai Custin (4.86, 1 h), Ford Bronco (4.86, 0 h)

### LARGE (1037 h, 40 models)

Hyundai Genesis (4.90, 3 h), Volkswagen Transporter (4.91, 1 h), Ford Maverick (4.93, 18 h), Hyundai Santa Cruz (4.94, 5 h), Genesis G80 (4.95, 3 h), Honda Pilot (4.97, 17 h), Genesis GV80 (4.97, 10 h), Acura MDX (4.97, 39 h), Tesla Model S (4.97, 7 h), Hyundai Ioniq 5 (5.00, 2 h), Honda Passport (5.02, 1 h), Tesla Model X (5.06, 16 h), Hyundai Palisade (5.08, 48 h), Lexus LS (5.11, 1 h), Volkswagen Atlas (5.11, 10 h), Ford Explorer (5.14, 16 h), Honda Odyssey (5.14, 67 h), Land Rover Defender (5.15, 0 h), Mazda CX-9 (5.15, 2 h), Genesis G90 (5.23, 6 h), Chevrolet Traverse (5.24, 1 h), Toyota Sienna (5.25, 7 h), Rivian R1S (5.26, 4 h), Kia EV9 (5.30, 16 h), Ford Ranger (5.30, 2 h), Kia Carnival (5.30, 27 h), Honda Ridgeline (5.33, 11 h), Hyundai Ioniq 9 (5.35, 2 h), Chrysler Pacifica (5.36, 14 h), GMC Yukon (5.45, 7 h), Ford Expedition (5.53, 6 h), Rivian R1T/R1S (5.58, 406 h), Cadillac Escalade ESV (5.76, 1 h), Ford Transit (5.83, 1 h), Volkswagen Crafter (6.24, 0 h), Ford F-150 Lightning (6.41, 8 h), Ford F-150 (6.44, 68 h), RAM 2500/3500 HD (6.57, 68 h), Chevrolet Silverado 1500 (6.65, 101 h), RAM 1500 (6.73, 15 h)

## Changing the thresholds

Edit the thresholds, re-run `scripts/s03c_lock_thresholds.py --write-config`, then `s03_labels.py`,
`s05_build_drives.py --relabel`, `s06_windows.py`, `s07b_apply_exclusions.py`, `s08_splits.py` and every experiment.
