# External vehicle specifications for the transfer test

`specs_external.csv` (99 rows, built 9 Nov 2026 by `build_specs_external.py`) gives wheelbase, front/rear track,
footprint, size class, curb mass and powertrain for every external vehicle model that the adapters of the zero-shot
campaign need: the phase-1 open sets (ULTra-AV, OpenACC, A2D2, Yuan et al. Tesla, microSIM-ACC/EV, UGA EV platoon,
CANdid, VED), the owner-download sets named in `CANDIDATES.md` sections 1-2 (commaCarSegments, commaSteeringControl,
MoreTorque, comma2k19, Ford Multi-AV), and every commaCarSegments key that is not in the lab table after opendbc
`MIGRATION` folding. Rebuild with `python3 build_specs_external.py` (stdlib only; prints a summary).

## Method

1. **Row = the model generation that actually appears in the external data.** Where the lab table
   (`vtt/specs_data/platform_specs.csv`) has that generation, its wheelbase, tracks and mass are copied verbatim and
   `source_url` reads `VTT table: <platform>` (42 rows, `spec_source = VTT table`). Where the external car is another
   generation than the lab row (Audi A4 B9 vs B8, Mazda CX-5 KE vs KF, Subaru Impreza 2012 vs 2017, Toyota Yaris XP130,
   Nissan Leaf ZE0 vs ZE1, VW Polo Mk5 vs Mk6, Acura RDX 2nd vs 3rd gen, Kia Sorento UM vs MQ4) the external generation
   is looked up and the lab row's numbers and class are quoted in `note` (`vtt_platform` names the related lab row).
2. **Web lookups (57 rows).** Manufacturer sheets first (Jaguar I-PACE technical data, Mazda USA CX-9 spec deck,
   Mazda Canada CX-50 sheet, SEAT Ateca technical specifications, Acura RDX fact sheet, US DOE AVT fact sheet for the
   2013 Leaf, Holden TM product information), otherwise the spec aggregators auto-data.net / cars-data.com /
   carspecs.us / cararac / thecarspec, the Australian listing sites carsales / carexpert (Redbook data) for the
   Australian CANdid cars and Omoda, and US listing sites (JD Power, cars.com, carbuzz, autobytel) for US-market
   cars. Every row cites one page; where the page gives a track range by wheel size the mid-point is used and the
   range is quoted in `note`. The OpenACC vehicle-specification CSVs of all seven campaigns (power, fuel, model year)
   were pulled from the JRC server to pin the variant (for example Cherasco's VW Polo is a 2010 Mk5, not the Mk6 in the
   lab table; the AstaZero Audi A6 is the 150 kW 40 TDI; the Casale follower is the Nexo).
3. **Footprint and class.** `footprint_m2 = wheelbase_m x (track_front + track_rear) / 2` in m, rounded to 0.001;
   SMALL < 4.43 <= MID < 4.88 <= LARGE (the locked thresholds of `docs/SIZE_CLASSES.md`, checked against
   `vtt/config.py` at build time).
4. **Borderline** = `|footprint - cut| < 0.05 m^2`, or an alternative published set of dimensions for the same car
   falls on the other side of a cut (two rows: Hyundai Ioniq 5 and Mazda CX-9, see conflicts). This is compatible
   with `common.validate_windows`, which OR-s an adapter-supplied flag with the strict 0.05 rule.
5. **Powertrain** uses the vocabulary of `common.POWERTRAINS` (BEV / ICE / HEV / PHEV / FCEV / unknown) and the lab
   conventions: 12 V / 48 V mild hybrids are ICE (as RAM eTorque, Focus MHEV, Defender MHEV in training) and carry
   the token `MHEV` in `note`; the BMW i3s REx (series range extender) is PHEV; platforms that merge gasoline and
   hybrid variants without a tag (Toyota Avalon TSS2, Honda Freed) and cars whose variant the dataset does not name
   (CATS Lincoln MKZ, Ohio Ford Fusion) are `unknown`. The two OpenACC cars whose spec file says HEV but which are
   mild hybrids (Mercedes GLE 450 EQ Boost, Audi A4 Avant 45 TFSI) are recorded as ICE with the exclusion of
   `CANDIDATES.md` quoted in `note`; keep them out of the powertrain score unless the mild-hybrid convention is
   applied explicitly.
6. **Confidence**: `high` = manufacturer or government document, or a lab-table row graded A/B; `medium` = spec
   aggregator or listing page (lab rows graded C, or web rows whose numbers were cross-checked against at least one
   other listing but not against a manufacturer sheet). No row is `low`; the rows that would deserve it were
   resolved by fetching a second page.

### Columns

`model, generation_or_years, wheelbase_mm, track_front_mm, track_rear_mm, footprint_m2, size_class, borderline,
curb_mass_kg, powertrain, source_url, confidence, note` as requested, plus four helper columns at the end:
`datasets` (semicolon list of the external sets the row serves), `vtt_platform` (the lab platform copied or
compared), `spec_source` (`VTT table` or `web`) and `platform_keys` (the opendbc / commaCarSegments /
commaSteeringControl keys named in the generation text, for joining by key). `model` alone is not unique (Mazda 3,
Mazda CX-5, Nissan Leaf, Kia Niro EV have one row per generation); join on `(model, generation_or_years)`,
`platform_keys` or `datasets`.

## Coverage

| dataset | rows | content |
|---|---|---|
| ULTra-AV / OpenACC | 28 | every named follower and leader of the seven JRC campaigns (3 Tesla, Audi A6 / A8 / A4 Avant / e-tron, BMW X5, Mercedes A-Class / GLE 450, Jaguar I-Pace, VW e-Golf / Polo, Mazda 3, Ford S-Max, Peugeot 3008 and 5008, Toyota Corolla, Hyundai Nexo / Ioniq Hybrid, Volvo XC40, Fiat 500X, Mitsubishi Space Star / Outlander PHEV, Mini Cooper, SsangYong Rexton, BMW i3s REx, Toyota RAV4 Hybrid, Kia Niro PHEV) plus the CATS Lincoln MKZ and the Ohio Ford Fusion |
| A2D2 | 1 | Audi Q7 e-tron (PHEV, size probe only) |
| Yuan et al. 2024 | 1 | Tesla Model 3 (2019 SR+) |
| microSIM-ACC/EV | 3 | Hyundai Ioniq 5, Polestar 2, Tesla Model 3 |
| UGA EV platoon | 3 | Lexus UX 200, Ford Mustang Mach-E, Kia Niro EV (2nd gen) |
| CANdid | 7 | Holden Astra / Barina / Cruze, Mazda CX-3 / CX-5 (KE), Subaru Impreza (2012), Toyota Yaris (XP130) |
| VED | 1 | Nissan Leaf ZE0 (2013) |
| comma2k19 | 2 | Honda Civic 2016, Toyota RAV4 2017 |
| MoreTorque/rlogs | 9 | Mazda CX-9 / 6 / CX-5 / CX-30 / CX-50 / 3 (BM and BP), Hyundai Sonata, Lexus RX, Omoda E5 |
| commaSteeringControl | 17 | the BEV archives and ICE archives named in CANDIDATES 4.2 |
| commaCarSegments | 21 + 24 | the named absent makes (Rivian R1 Gen1, Honda e, BYD Atto 3, Peugeot 208, MG5 EV) and, as tier 2, all 24 further keys that remain outside the lab table after MIGRATION (Lexus NX, Skoda Octavia III, Genesis G80 DH, Toyota Mirai, Toyota Avalon TSS2, SEAT Ateca, VW T-Roc, Acura RDX 2G, Kia Stinger 2022, Ioniq PHEV 2019, Elantra GT / i30, Lexus CT 200h, Staria, Lexus RC, Prius v, Buick Regal, GMC Acadia, Honda Freed, Skoda Kamiq, Dodge Durango, Perodua Myvi, Kia Sorento UM, Cadillac Escalade ESV / Escalade) |
| Ford Multi-AV | 1 | Ford Fusion Hybrid (HEV, size probe only) |

Class and powertrain counts over the 99 rows: SMALL 56 / MID 26 / LARGE 17; BEV 23, ICE 60, HEV 5, PHEV 5, FCEV 2,
unknown 4. Confidence: high 34, medium 65.

## Borderline models (flag `borderline = true`)

| model | footprint | class | why |
|---|---|---|---|
| Audi A8 (D5) | 4.912 | LARGE | 0.032 above the cut (leader only in AstaZero) |
| Audi A4 Avant (B9) | 4.409 | SMALL | 0.021 below the cut; lab B8 row 4.373 |
| Audi e-tron | 4.841 | MID | 0.039 below the cut |
| Jaguar I-Pace | 4.917 | LARGE | wheel-size track range gives 4.89-4.94 |
| Hyundai Ioniq 5 | 4.826 | MID | lab value kept (0.054 below the cut); public 3000 mm wheelbase gives 4.91 LARGE |
| Polestar 2 | 4.380 | SMALL | 0.050 below the cut |
| Ford Mustang Mach-E | 4.845 | MID | 0.035 below the cut |
| Mazda CX-9 | 4.856 | MID | published 2930 mm wheelbase; lab row 3100 mm gives 5.149 LARGE |
| Mazda 6 | 4.471 | MID | 0.041 above the cut |
| Toyota Camry (XV70) | 4.468 | MID | 0.038 above the cut |

(Toyota Avalon TSS2 at 4.526 and Lincoln MKZ at 4.499 are the next closest, 0.07-0.10 from a cut, not flagged.)

## Conflicts resolved

* **Hyundai Ioniq 5**: lab table 2970 x 1625 mm = 4.826 MID; Hyundai's press kit gives a 3000 mm wheelbase and the
  aggregators 1628-1638 / 1637-1647 mm tracks = 4.91 LARGE. The lab value is kept, as ruled in `CANDIDATES.md`
  (the training labels used it), and the row is flagged borderline so that the size metrics are also reported
  without it. The lab row itself should be re-checked (see below).
* **Mazda CX-9**: lab row MAZDA_CX9_2021 has a 3100 mm wheelbase (5.149 LARGE); Mazda USA's 2021 spec deck gives
  115.3 in = 2930 mm and tracks 65.3 / 65.2 in = 4.856 MID (0.024 below the cut). The published value is used,
  flagged borderline, and the note asks to report both labels or exclude the CX-9 from the size score.
* **Peugeot 3008 vs 5008**: both rows are given. The CSV headers of the Vicolungo files say 3008 GT Line (4.265
  SMALL), the JRC spec file says 5008 GT Line (4.538 MID); the class flips with the identity, so the car stays
  excluded as `CANDIDATES.md` decided.
* **VW Polo**: the Cherasco leader is a 2010 car (JRC spec file: 63 kW, 1390 cc, petrol/LPG) = Polo Mk5 6R
  (3.605 SMALL), not the Mk6 of the lab table (3.863, same class).
* **Casale ids**: the ULTra-AV README lists "Rexton, Hyundai" while the JRC spec file lists Nexo and Rexton;
  `CANDIDATES.md` found the CSV ids inverted (id 0 = Nexo). Both cars have rows.
* **Mild hybrids labelled HEV by OpenACC** (GLE 450 EQ Boost, A4 Avant 45 TFSI): recorded ICE with `MHEV` in the
  note and the exclusion quoted; A6 40 TDI, A8 50 TDI and Mazda 3 BP (24 V M Hybrid) are the same kind of car and
  OpenACC labels them ICE, so the exclusion is a choice of `CANDIDATES.md`, not a spec question.
* **BMW i3s "HEV"**: a range-extended EV, classed PHEV here; outside the task either way.
* **Lincoln MKZ / Ohio Ford Fusion**: neither the ULTra-AV paper nor the FHWA report names the powertrain variant;
  `powertrain = unknown`. Both are outside the scored sets in `CANDIDATES.md` 2.3.
* **MG5 EV**: the lab row has wheelbase 2659 and mass 1637 but no tracks; the complete auto-data set (2665 /
  1558 / 1553, 1558 kg) is used; footprint 4.14 either way.
* **Jaguar I-Pace**: tracks vary with wheel size (1624-1643 / 1647-1663 mm in Jaguar's sheet); mid-points give
  4.917, the range 4.89-4.94, all LARGE but flagged.

## Not sourced (no row)

* Vehicles the datasets do not identify: the Ohio "retrofitted Tesla sedan" (ULTra-AV id 0 of both Ohio sets), the
  Vanderbilt 2019 SUV, the ZalaZone Smart and Skoda target leaders, the unnamed BMW leader of the JRC low-speed
  campaign, the Waymo / Argoverse vehicles (not scored), the VED fleet apart from the three Leafs (make/model
  withheld).
* commaCarSegments keys that are not cars or not identifiable: `TURBO_RC_CAR` (3 segments); `HONDA_E` is the same
  car as the lab row HONDA_E_ADVANCE and is covered by the Honda e row. The "Tesla 2025 port" of CANDIDATES 2.1 is
  the TESLA_MODEL_3 / TESLA_MODEL_Y platform pair, which the lab table already holds (Model 3 row here; Model Y is in
  the lab table).
* Powertrain could not be fixed for four rows (`unknown`): Lincoln MKZ, Ohio Ford Fusion, Toyota Avalon TSS2
  (gasoline and hybrid on one platform; the lab Toyota ECU rule can split them), Honda Freed (petrol and e:HEV on
  one fingerprint). Peugeot 208 is recorded ICE but the opendbc platform also covers the e-208.

## Suggested corrections to the lab table (not applied here; `vtt/specs_data` is left untouched)

* `MAZDA_CX9_2021`: wheelbase 3100 -> 2930 mm (Mazda USA spec deck); the current value looks like the CX-90's
  3120 mm. Changes the lab class from LARGE to MID (borderline).
* `HYUNDAI_IONIQ_5`: wheelbase 2970 -> 3000 mm (Hyundai press kit; HYUNDAI_IONIQ_5_PE and _N already have 3000).
  Would move the model from MID (4.826) to LARGE (4.875 with the table's 1625 mm tracks, 4.91 with public tracks),
  which touches 110 h of training data and the locked thresholds, so this is a user decision, not an edit.
* `MG_5_EV`: tracks 1558 / 1553 mm are missing from the table.
* `KIA_NIRO_EV_2ND_GEN`: wheelbase 2700 -> 2720 mm (same class).

## Verification material

Downloaded references (untrusted, kept apart from code): `<data-root>/scratch/specs_ext_verify/`
holds the JRC spec and notes files of all seven OpenACC campaigns (`jrc/`), the UGA README (`readmes/`) and the
manufacturer PDFs that were read with `pdftotext` (`pdfs/`: Jaguar I-PACE 22MY, Mazda CX-9 2021 deck, Mazda CX-50
2024 Canada, Omoda E5 Australia). The microSIM-ACC/EV README is no longer served by GitHub (404); the vehicle list of
`CANDIDATES.md` (Ioniq 5 2022, Polestar 2 2023, Model 3 2022) was used.
