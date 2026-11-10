#!/usr/bin/env python3
"""Build specs_external.csv: one row per external vehicle model (generation) that appears in the phase-1 external
datasets or in the commaCarSegments / commaSteeringControl candidate lists of CANDIDATES.md.

Where the lab table (vtt/specs_data/platform_specs.csv) has the same model generation its values are copied verbatim
(source 'VTT table: <platform>'); otherwise the numbers were looked up on the web (manufacturer sheets preferred,
reputable spec aggregators otherwise) and the page is cited in source_url. Footprint, size class and the borderline
flag are computed here with the locked thresholds (4.43 / 4.88 m^2, docs/SIZE_CLASSES.md).

borderline = |footprint - cut| < 0.05 m^2, OR an alternative published set of dimensions for the same car puts it on
the other side of a cut (flag `alt_cross` below; the conflict is spelled out in `note`). This is compatible with
common.validate_windows, which OR-s an adapter-supplied flag with the strict 0.05 rule.

Stdlib only. Run:  python3 build_specs_external.py  (writes specs_external.csv next to this file and prints a summary).
No device or route identifiers are read or written.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLATFORM_SPECS = ROOT / "vtt" / "specs_data" / "platform_specs.csv"
OUT = HERE / "specs_external.csv"

CUTS = (4.43, 4.88)          # docs/SIZE_CLASSES.md (locked 6 Nov 2026); checked against vtt/config.py below
BORDERLINE_M2 = 0.05
COLUMNS = ["model", "generation_or_years", "wheelbase_mm", "track_front_mm", "track_rear_mm", "footprint_m2",
           "size_class", "borderline", "curb_mass_kg", "powertrain", "source_url", "confidence", "note",
           # extra columns (after the required ones): where the model appears, the related lab platform, data origin
           "datasets", "vtt_platform", "spec_source", "platform_keys"]
POWERTRAINS = {"BEV", "ICE", "HEV", "PHEV", "FCEV", "unknown"}
VTT_CONF = {"A": "high", "B": "high", "C": "medium"}


def check_thresholds() -> None:
    cfg = ROOT / "vtt" / "config.py"
    if cfg.exists():
        txt = cfg.read_text()
        if "(4.43, 4.88)" not in txt:
            sys.exit("vtt/config.py FOOTPRINT_THRESHOLDS_M2 differs from the (4.43, 4.88) hard-coded here")


_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9-]+)+\b|'([A-Z][A-Z0-9 ]+)'")


def platform_keys(gen: str, vtt_platform: str) -> str:
    """Best-effort list of platform / archive keys (UPPER_SNAKE tokens and quoted legacy keys in the generation text, plus the lab platform)."""
    keys = []
    if vtt_platform:
        keys.append(vtt_platform)
    for m in _KEY_RE.finditer(gen):
        k = m.group(1) or m.group(0)
        if k not in keys and not k.isdigit():
            keys.append(k)
    return "; ".join(keys)


def footprint(wb: float, tf: float, tr: float) -> float:
    return round(wb / 1000.0 * ((tf + tr) / 2.0) / 1000.0, 3)


def size_class(fp: float) -> str:
    return "SMALL" if fp < CUTS[0] else ("MID" if fp < CUTS[1] else "LARGE")


def is_borderline(fp: float) -> bool:
    return any(abs(fp - c) < BORDERLINE_M2 for c in CUTS)


_HEX16 = re.compile(r"[0-9a-fA-F]{16,}")


def clean_url(urls: list[str]) -> str:
    """First URL without a long hex run (commit hashes, press-release ids), '' if none."""
    for u in urls:
        if not _HEX16.search(u):
            return u
    return ""


def load_vtt() -> dict[str, dict]:
    with open(PLATFORM_SPECS, newline="") as f:
        return {r["platform"]: r for r in csv.DictReader(f)}


# ------------------------------------------------------------------------------------------------- rows
# V(...) = copy the lab table; W(...) = web lookup. Fields: model, gen, [wb, tf, tr, mass], pt, url, conf, note,
# datasets, vtt_platform, alt_cross.
def V(model, gen, platform, pt, datasets, note="", alt_cross=False, mass=None):
    return dict(kind="vtt", model=model, gen=gen, platform=platform, pt=pt, datasets=datasets, note=note,
                alt_cross=alt_cross, mass=mass)


def W(model, gen, wb, tf, tr, mass, pt, url, conf, datasets, note="", vtt_platform="", alt_cross=False):
    return dict(kind="web", model=model, gen=gen, wb=wb, tf=tf, tr=tr, mass=mass, pt=pt, url=url, conf=conf,
                datasets=datasets, note=note, vtt_platform=vtt_platform, alt_cross=alt_cross)


MHEV = "mild hybrid (MHEV) counted as ICE by the lab convention (RAM eTorque, Focus MHEV, Defender MHEV = ICE)"

ROWS = [
    # ---------------------------------------------------------------- ULTra-AV / OpenACC (CANDIDATES 2.3, 2.4)
    V("Tesla Model 3", "2017-2023 pre-facelift (AstaZero 2019, ZalaZone 2019, Yuan 2019 SR+ RWD, microSIM 2022)",
      "TESLA_MODEL_3", "BEV", "ULTra-AV; OpenACC; Yuan2024; microSIM-EV",
      "all VTT Model 3 platforms share these dimensions"),
    V("Tesla Model S", "2016-2020 pre-refresh (ZalaZone 2018, 75 kWh)", "TESLA_MODEL_S_HW3", "BEV", "ULTra-AV; OpenACC",
      "all VTT Model S platforms (PREAP/HW1/HW2/HW3/RAVEN) share these dimensions"),
    V("Tesla Model X", "2016-2020 pre-refresh (ZalaZone 2016, 90 kWh)", "TESLA_MODEL_X", "BEV", "ULTra-AV; OpenACC"),
    W("Audi A6", "C8 2018 (AstaZero follower: 150 kW diesel = 40 TDI)", 2924, 1630, 1617, 1710, "ICE",
      "https://www.auto-data.net/en/audi-a6-avant-c8-40-tdi-204hp-mhev-s-tronic-33868", "medium",
      "ULTra-AV; OpenACC", f"40 TDI has a 12 V {MHEV}; tracks identical for saloon and Avant; kerb 1710 kg FWD (quattro ~1770)"),
    W("Audi A8", "D5 2018 standard wheelbase (AstaZero leader: 210 kW diesel = 50 TDI)", 2998, 1644, 1633, 1975, "ICE",
      "https://www.auto-data.net/en/audi-a8-d5-50-tdi-v6-286hp-mild-hybrid-quattro-tiptronic-31522", "medium",
      "OpenACC", f"48 V {MHEV}; if the car were an A8 L (3128 mm) the footprint would be 5.13 (still LARGE); leader only"),
    W("Audi A4 Avant", "B9 facelift 2019 (ZalaZone: 140 kW petrol = 45 TFSI quattro MHEV)", 2820, 1572, 1555, 1590, "ICE",
      "https://www.auto-data.net/en/audi-a4-avant-b9-8w-facelift-2019-45-tfsi-245hp-quattro-mild-hybrid-s-tronic-37071",
      "medium", "ULTra-AV; OpenACC",
      f"12 V {MHEV}; OpenACC spec file lists it HEV and CANDIDATES.md excludes it from the powertrain score; "
      "VTT table AUDI_A4_4TH_GEN is the previous B8 generation (2808/1564/1551 -> 4.373 SMALL, same class)",
      vtt_platform="AUDI_A4_4TH_GEN"),
    W("Audi e-tron", "2019 (55 quattro, 95 kWh; later Q8 e-tron)", 2928, 1655, 1652, 2490, "BEV",
      "https://www.auto-data.net/en/audi-e-tron-55-95-kwh-408hp-quattro-34272", "medium", "ULTra-AV; OpenACC",
      "0.04 below the LARGE cut; rear track 1652-1655 depending on listing"),
    W("BMW X5", "G05 2018 (AstaZero: 195 kW diesel = xDrive30d 265 hp)", 2975, 1666, 1685, 2110, "ICE",
      "https://www.auto-data.net/en/bmw-x5-g05-30d-265hp-xdrive-steptronic-33266", "medium", "ULTra-AV; OpenACC",
      "2018 30d has no 48 V system (added with the 286 hp 2020 version); kerb 2110-2185 kg by seat count"),
    W("Mercedes-Benz A-Class", "W177 2019 (AstaZero: 165 kW petrol = A 250)", 2729, 1567, 1547, 1380, "ICE",
      "https://www.auto-data.net/en/mercedes-benz-a-class-w177-a-250-224hp-7g-dct-32806", "medium", "ULTra-AV; OpenACC"),
    W("Mercedes-Benz GLE 450 4MATIC", "V167 2019 (ZalaZone: 270 kW petrol EQ Boost)", 2995, 1667, 1687, 2145, "ICE",
      "https://www.auto-data.net/en/mercedes-benz-gle-suv-v167-gle-450-367hp-eq-boost-4matic-g-tronic-35268", "medium",
      "ULTra-AV; OpenACC",
      f"48 V EQ Boost {MHEV}; OpenACC spec file lists it HEV and CANDIDATES.md excludes it from the powertrain score"),
    W("Jaguar I-Pace", "2018-2023 (EV400, 90 kWh; ZalaZone 2019)", 2990, 1634, 1655, 2133, "BEV",
      "https://cdn.uc.assets.prezly.com/398b39e5-7089-402e-b6e4-29212bb0d097/-/inline/no/Jaguar%20I-PACE%2022MY%20Technical%20Data.pdf",
      "high", "ULTra-AV; OpenACC",
      "Jaguar 22MY technical data: wheelbase 2990, front track 1624-1643, rear 1647-1663 by wheel size (mid-points used), "
      "DIN 2133 kg; auto-data 1642/1661 gives 4.94; footprint range 4.89-4.94 -> LARGE but within 0.05 of the cut"),
    V("Volkswagen e-Golf", "Mk7 2014-2020 (Vicolungo 2018, 35.8 kWh)", "VOLKSWAGEN_GOLF_MK7", "BEV", "ULTra-AV; OpenACC",
      "VTT platform row covers all Golf Mk7 variants; the e-Golf kerb weight is about 1615 kg (public), the table's "
      "1397 kg is the platform value"),
    W("Volkswagen Polo", "Mk5 6R 2009-2014 (Cherasco leader: 2010, 1.4 63 kW petrol/LPG)", 2470, 1463, 1456, 1004, "ICE",
      "https://cars-data.com/en/volkswagen/polo/6r-2009-hatchback-8082/1-4-highline-53267--53267/specs", "medium",
      "OpenACC", "VTT table VOLKSWAGEN_POLO_MK6 is the 2018+ generation (2550/1525/1505 -> 3.863 SMALL, same class); "
      "leader only (no lead features)", vtt_platform="VOLKSWAGEN_POLO_MK6"),
    W("Mazda 3", "BP 2019- (ZalaZone 2019, 2.0 Skyactiv-G petrol)", 2725, 1570, 1580, 1249, "ICE",
      "https://cars-data.com/en/mazda/3/bp-2019-hatchback/skyactiv-g-2-0-122-96395--96395/specs", "medium",
      "ULTra-AV; OpenACC", f"EU 2.0 Skyactiv-G carries a 24 V M Hybrid {MHEV}; kerb 1249 (EU 122 hp) to 1317 kg (AU G20)"),
    W("Ford S-Max", "Mk2 2015-2023 (Vicolungo 2018 and JRC low speed: 110 kW diesel = 2.0 TDCi 150)", 2849, 1606, 1606, 1725,
      "ICE", "https://www.auto-data.net/en/ford-s-max-ii-2.0-tdci-150hp-s-s-21477", "medium", "ULTra-AV; OpenACC"),
    W("Peugeot 3008", "II 2017-2020 (Vicolungo CSV headers: 3008 GT Line, 130 kW diesel)", 2675, 1590, 1599, 1480, "ICE",
      "https://www.auto-data.net/en/peugeot-3008-ii-phase-i-2016-gt-2.0-bluehdi-177hp-automatic-s-s-34445", "medium",
      "ULTra-AV; OpenACC",
      "tracks are listed as ranges 1579-1601 / 1587-1610 (wheel size); mid-points used; footprint 4.23-4.30 all SMALL. "
      "The OpenACC spec file calls the same car a 5008 (next row, MID): identity unresolved -> excluded by CANDIDATES.md"),
    W("Peugeot 5008", "II 2017- (OpenACC Vicolungo spec file: 5008 GT Line 2.0 BlueHDi)", 2840, 1597, 1599, 1530, "ICE",
      "https://www.auto-data.net/en/peugeot-5008-ii-phase-i-2017-2.0-bluehdi-180hp-automatic-27390", "medium",
      "ULTra-AV; OpenACC",
      "tracks listed as ranges 1593-1601 / 1587-1610 (mid-points used); footprint 4.52-4.56 all MID. See the 3008 row: "
      "the class flips with the identity (3008 SMALL vs 5008 MID)"),
    V("Toyota Corolla", "E210 2020 (microSIM ICE: Corolla LE 2020, gasoline)", "TOYOTA_COROLLA_TSS2", "ICE", "ULTra-AV; commaSteeringControl",
      "platform row merges sedan/hatch (wheelbase 2700/2640 -> table 2670); TSS2 platform also covers the hybrid, the microSIM car is the gasoline LE"),
    V("Hyundai Nexo", "2019 (Casale id 0 and JRC low speed; fuel cell)", "HYUNDAI_NEXO_1ST_GEN", "FCEV", "ULTra-AV; OpenACC",
      "FCEV: outside the BEV-vs-ICE task"),
    V("Hyundai Ioniq Hybrid", "AE 2017-2019 (Cherasco follower 2018, 104 kW hybrid)", "HYUNDAI_IONIQ", "HEV", "OpenACC"),
    W("Volvo XC40", "2018 (Cherasco follower: 140 kW diesel = D4 AWD)", 2702, 1601, 1626, 1633, "ICE",
      "https://cars-data.com/en/volvo/xc40/2018/d4-awd-r-design--92840/specs", "medium", "OpenACC",
      "tracks agree across cars-data, auto-data and carexpert; kerb 1610-1728 kg by variant"),
    W("Fiat 500X", "2014-2018 (one-vehicle campaign 2016-17 and Cherasco leader: 103 kW diesel = 2.0 MultiJet 140)", 2570, 1545, 1545, 1495,
      "ICE", "https://www.auto-data.net/en/fiat-500x-2.0-140hp-awd-automatic-22098", "medium", "OpenACC",
      "Stellantis technical sheet and thecarspec agree on 2570 / 1545 / 1545; kerb 1430-1570 kg by gearbox/drive"),
    W("Mitsubishi Space Star", "2016-2020 facelift (Vicolungo 2018, 1.2 59 kW petrol; Mirage outside Europe)", 2450, 1430, 1415, 880, "ICE",
      "https://cars-data.com/en/mitsubishi/space-star/2016/1-2-instyle-96459--96459/specs", "medium", "ULTra-AV; OpenACC"),
    W("Mitsubishi Outlander PHEV", "III GF/GG 2018 (Vicolungo 2018, 2.4 PHEV)", 2670, 1540, 1540, 1880, "PHEV",
      "https://www.auto-data.net/en/mitsubishi-outlander-iii-facelift-2018-2.4-mivec-224hp-plug-in-hybrid-4wd-33489", "medium",
      "ULTra-AV; OpenACC", "PHEV: outside the BEV-vs-ICE task"),
    W("MINI Cooper", "F56 3-door facelift 2018 (Vicolungo 2018, 1.5 100 kW petrol)", 2495, 1501, 1501, 1135, "ICE",
      "https://www.auto-data.net/en/mini-hatch-f56-facelift-2018-3-door-cooper-1.5-136hp-34117", "medium", "ULTra-AV; OpenACC",
      "body (3-door F56 vs 5-door F55, wheelbase 2567 -> 3.85) not stated by OpenACC; SMALL either way"),
    W("SsangYong Rexton", "Y400 2017- (Casale id 1: 2019, 2.2 diesel 133 kW)", 2865, 1640, 1640, 2130, "ICE",
      "https://www.carexpert.co.nz/kgm/rexton/2021/exterior-and-dimensions", "medium", "ULTra-AV; OpenACC",
      "kerb 2060 kg RWD / 2130 kg 4x4 (drive of the test car not stated); ULTra-AV README inverts the Casale ids "
      "(CSV id 0 = Nexo, id 1 = Rexton per CANDIDATES.md)"),
    W("BMW i3s REx", "I01 120 Ah 2018 (ZalaZone: i3s with range extender, 647 cc petrol, 33.2 kWh)", 2570, 1592, 1578, 1385, "PHEV",
      "https://www.press.bmwgroup.com/latin-america-caribbean/article/attachment/T0285510EN/415638", "medium", "ULTra-AV; OpenACC",
      "series range-extended EV: classed PHEV here (OpenACC spec file says HEV); outside the BEV-vs-ICE task. "
      "Tracks from the BMW i3s (120 Ah) press data sheet, DIN weight of the REx version 1385 kg from the Nov-2017 BMW sheet / autodata1"),
    V("Toyota RAV4 Hybrid", "XA50 2019 (ZalaZone 2019, 2.5 hybrid)", "TOYOTA_RAV4_TSS2", "HEV", "ULTra-AV; OpenACC",
      "HEV: outside the BEV-vs-ICE task"),
    V("Kia Niro PHEV", "DE 2019 (Vicolungo 2019, 8.9 kWh plug-in)", "KIA_NIRO_PHEV", "PHEV", "ULTra-AV; OpenACC",
      "PHEV: outside the BEV-vs-ICE task"),
    W("Lincoln MKZ", "2nd gen 2013-2020 (CATS Lab 2016 black and 2017 red research cars)", 2850, 1582, 1575, 1756, "unknown",
      "https://www.carspecs.us/cars/2017/lincoln/mkz/68738", "medium", "ULTra-AV",
      "tracks 62.3 / 62.0 in; curb 3871 lb (Hybrid) or 3739 lb (2.0T, 1696 kg). Powertrain of the CATS cars not stated in the "
      "ULTra-AV paper (CANDIDATES.md assumes hybrid): excluded from the powertrain score either way"),
    W("Ford Fusion", "2nd gen 2013-2020 (Ohio two-vehicle SV2, manual driving, model year not stated)", 2850, 1593, 1585, 1556, "unknown",
      "https://www.jdpower.com/cars/2014/ford/fusion/sedan-4d-se-i4/specs", "medium", "ULTra-AV",
      "tracks 62.7 / 62.4 in, curb 3431 lb (SE 2.5); the FHWA report does not say whether the Ohio car is the hybrid"),
    W("Ford Fusion Hybrid", "2nd gen 2013-2020 (Ford Multi-AV Seasonal 2014 test cars; Argoverse)", 2850, 1593, 1585, 1640, "HEV",
      "https://www.cars.com/research/ford-fusion_hybrid-2014/specs/360664", "medium", "FordMultiAV; ULTra-AV",
      "same body as the Fusion; curb 3615 lb; HEV: outside the BEV-vs-ICE task; Ford Multi-AV is an OWNER download of lowest priority"),
    # ---------------------------------------------------------------- A2D2
    W("Audi Q7 e-tron quattro", "4M 2016-2018 (3.0 TDI diesel plug-in hybrid)", 2994, 1679, 1691, 2445, "PHEV",
      "https://www.auto-data.net/en/audi-q7-ii-typ-4m-e-tron-3.0-tdi-374hp-plug-in-hybrid-quattro-tiptronic-23166", "medium",
      "A2D2", "autodata1 lists the front track as 1663 (4.99 with the same rear track): LARGE either way; PHEV -> size probe only"),
    # ---------------------------------------------------------------- microSIM-ACC/EV, UGA
    V("Hyundai Ioniq 5", "NE 2022 (microSIM 2022; commaSteeringControl HYUNDAI_IONIQ_5_2022)", "HYUNDAI_IONIQ_5", "BEV",
      "microSIM-EV; commaSteeringControl; commaCarSegments",
      "CONFLICT: lab table 2970 x 1625 -> 4.826 MID (0.054 below the cut); public specs give wheelbase 3000 mm (Hyundai press kit) "
      "and tracks 1628-1638 / 1637-1647 (auto-data) -> 4.91 LARGE. Lab value kept per CANDIDATES.md (training labels used it); "
      "flagged borderline because the public dimensions cross the cut", alt_cross=True),
    V("Polestar 2", "2020- (microSIM 2023)", "POLESTAR_2", "BEV", "microSIM-EV", "0.050 below the MID cut"),
    W("Lexus UX 200", "ZA10 2019- (UGA lead car 2021, 2.0 petrol)", 2640, 1550, 1550, 1435, "ICE",
      "https://cars-data.com/en/lexus/ux/2019/200-preference-line--103364/specs", "medium", "UGA",
      "US curb 3307 lb (1500 kg); TNGA-C platform shared with the Corolla; lead car only (no lead features)"),
    V("Ford Mustang Mach-E", "Mk1 2021- (UGA 2024)", "FORD_MUSTANG_MACH_E_MK1", "BEV", "UGA",
      "0.035 below the LARGE cut; VTT docs list 2021-23, the 2024 car is the same generation"),
    V("Kia Niro EV", "SG2 2nd gen 2023 (UGA, two cars)", "KIA_NIRO_EV_2ND_GEN", "BEV", "UGA",
      "public wheelbase 2720 mm (table 2700) -> 4.29, same class"),
    # ---------------------------------------------------------------- CANdid (Adelaide)
    W("Holden Astra", "BK hatch 2018 (CANdid car 1, 1.4 turbo petrol; = Opel Astra K)", 2662, 1574, 1581, 1283, "ICE",
      "https://www.carsales.com.au/research/holden/astra/2018/r-plus/", "medium", "CANdid", "kerb 1283-1363 kg by variant"),
    W("Holden Barina", "TM hatch 2012 (CANdid cars 4 auto and 10 manual, 1.6 petrol; = Chevrolet Sonic/Aveo T300)", 2525, 1497, 1495, 1256, "ICE",
      "https://australiancar.reviews/_pdfs/Holden_Barina_TM_ProductInformation_201109.pdf", "medium", "CANdid",
      "Holden TM product information: 1229 kg manual / 1256 kg automatic; carsales sedan listing agrees on 2525 / 1497 / 1495"),
    W("Holden Cruze", "JH sedan 2012 (CANdid car 5, petrol auto; = Chevrolet Cruze J300)", 2685, 1541, 1563, 1373, "ICE",
      "https://www.carsales.com.au/research/holden/cruze/2012/cd/", "medium", "CANdid",
      "tracks 1539-1545 / 1558-1567 by variant; kerb 1373 kg petrol sedan, 1549 kg 2.0 diesel"),
    W("Mazda CX-3", "DK 2017 (CANdid car 2, 2.0 petrol FWD)", 2570, 1525, 1520, 1251, "ICE",
      "https://www.carsales.com.au/research/mazda/cx-3/2017/maxx", "medium", "CANdid", "kerb 1219-1356 kg by drive/variant"),
    W("Mazda CX-5", "KE 1st gen 2013-2014 (CANdid cars 7 and 8, petrol)", 2700, 1585, 1590, 1559, "ICE",
      "https://www.carsales.com.au/research/mazda/cx-5/2013/maxx/", "medium", "CANdid",
      "kerb 1475 kg FWD to 1688 kg diesel AWD; VTT table MAZDA_CX5 is the KF 2017-21 generation (2700/1595/1595 -> 4.306 SMALL, same class)",
      vtt_platform="MAZDA_CX5"),
    W("Subaru Impreza", "GJ/GP 4th gen 2012 (CANdid cars 3 and 6, 2.0 petrol)", 2645, 1510, 1515, 1375, "ICE",
      "https://www.carsales.com.au/research/subaru/impreza/2012/2-0i-l/", "medium", "CANdid",
      "VTT table SUBARU_IMPREZA is the 2017-19 generation (2670/1549/1545 -> 4.130 SMALL, same class)", vtt_platform="SUBARU_IMPREZA"),
    W("Toyota Yaris", "XP130 NCP130 hatch 2016 (CANdid car 9, 1.3 petrol)", 2510, 1475, 1460, 1035, "ICE",
      "https://www.carsales.com.au/research/toyota/yaris/2016/ascent/", "medium", "CANdid",
      "VTT table TOYOTA_YARIS (2560/1480/1475 -> 3.782) is a later generation; same class", vtt_platform="TOYOTA_YARIS"),
    # ---------------------------------------------------------------- commaCarSegments / CI routes: makes absent from the lab inventory
    V("Rivian R1T/R1S", "Gen 1 2022-2024 (commaCarSegments RIVIAN_R1_GEN1)", "RIVIAN_R1_GEN1", "BEV", "commaCarSegments; CI-routes",
      "shared platform row uses the mean wheelbase; R1T 3449 -> 5.90, R1S 3076 -> 5.26, all LARGE"),
    V("Honda e", "2020- (commaCarSegments HONDA_E / HONDA_E_ADVANCE, CI routes)", "HONDA_E_ADVANCE", "BEV", "commaCarSegments; CI-routes",
      "commaCarSegments key HONDA_E is not in the lab table; it is the same car as HONDA_E_ADVANCE"),
    V("BYD Atto 3", "2022- (commaCarSegments, CI routes)", "BYD_ATTO3", "BEV", "commaCarSegments; CI-routes"),
    V("Peugeot 208", "II 2019- (commaCarSegments PSA_PEUGEOT_208, CI routes)", "PSA_PEUGEOT_208", "ICE", "commaCarSegments; CI-routes",
      "opendbc platform also covers the e-208 (BEV); the powertrain of a given route is unknown without the car's own signals"),
    W("MG5 EV", "2020-2022 pre-facelift estate (commaCarSegments MG_5_EV, CI routes)", 2665, 1558, 1553, 1558, "BEV",
      "https://auto-data.net/en/mg-mg5-ev-long-range-61.1-kwh-156hp-44662", "medium", "commaCarSegments; CI-routes",
      "VTT table MG_5_EV has wheelbase 2659 and mass 1637 but no tracks, so the complete auto-data set is used "
      "(kerb 1550-1565 kg; UK sources give 2659); footprint 4.14 either way", vtt_platform="MG_5_EV"),
    # ---------------------------------------------------------------- VED
    W("Nissan Leaf", "ZE0 2013 (VED: three 2013 Leaf 24 kWh, the only BEVs)", 2700, 1539, 1534, 1493, "BEV",
      "https://www.energy.gov/sites/prod/files/2015/01/f19/fact2013nissanleaf.pdf", "high", "VED",
      "DOE AVT fact sheet: wheelbase 106.3 in, track 60.6 / 60.4 in; curb 3291 lb (SV); VTT NISSAN_LEAF is the ZE1 2018-23 "
      "(2705/1538/1538 -> 4.160 SMALL, same class)", vtt_platform="NISSAN_LEAF"),
    # ---------------------------------------------------------------- comma2k19
    V("Honda Civic", "10th gen 2016 Touring (comma2k19)", "HONDA_CIVIC", "ICE", "comma2k19; commaSteeringControl"),
    V("Toyota RAV4", "XA40 2017 Platinum (comma2k19)", "TOYOTA_RAV4", "ICE", "comma2k19"),
    # ---------------------------------------------------------------- MoreTorque/rlogs (OWNER download; CX-9 conflict assigned)
    W("Mazda CX-9", "TC 2nd gen 2016-2023 (MoreTorque MAZDA_CX9_2021)", 2930, 1659, 1656, 1921, "ICE",
      "https://news.mazdausa.com/download/2021+CX-9+Spec+Deck.pdf", "high", "MoreTorque",
      "CONFLICT: lab table MAZDA_CX9_2021 has wheelbase 3100 -> 5.149 LARGE (the CX-90 has 3120 mm); Mazda USA 2021 spec deck: "
      "wheelbase 115.3 in = 2930 mm, track 65.3 / 65.2 in, curb 4236 lb FWD / 4409 lb AWD -> 4.86 MID, 0.02 below the cut. "
      "Report both labels or exclude the CX-9 from the size score; the lab table row should be corrected", vtt_platform="MAZDA_CX9_2021",
      alt_cross=True),
    V("Mazda 6", "GJ/GL 2017-2020 (MoreTorque MAZDA_6)", "MAZDA_6", "ICE", "MoreTorque", "0.041 above the MID cut"),
    V("Mazda CX-5", "KF 2nd gen 2017-2021 and 2022-2024 (MoreTorque MAZDA_CX5 / MAZDA_CX5_2022)", "MAZDA_CX5", "ICE", "MoreTorque; commaCarSegments",
      "MAZDA_CX5_2022 has identical dimensions; Mazda 24 V mild hybrids follow the lab mild-hybrid convention (ICE)"),
    W("Mazda CX-30", "DM 2020- (MoreTorque community platform)", 2655, 1565, 1565, 1537, "ICE",
      "https://www.carspecs.us/cars/2020/mazda/cx-30/83060", "medium", "MoreTorque",
      "wheelbase 104.5 in, tracks 61.6 / 61.6 in, curb 3388 lb (Select AWD; FWD ~3232 lb)"),
    W("Mazda CX-50", "2023- (MoreTorque community platform)", 2815, 1658, 1662, 1717, "ICE",
      "https://en.media.mazda.ca/download/%28English%29+2024+Mazda+CX-50++Specifications+and+Features.pdf", "high", "MoreTorque",
      "Mazda Canada specification sheet: tracks 1658 / 1662 on 17-in wheels (1648 / 1652 on 20-in), curb 1717 kg (2.5) / 1777 kg (2.5 T)"),
    W("Mazda 3", "BM 3rd gen 2014-2018 (MoreTorque Mazda3 community platform)", 2700, 1555, 1560, 1308, "ICE",
      "https://cararac.com/spec/mazda/3/hatchback-2013/", "medium", "MoreTorque",
      "curb 1190-1330 kg by engine (cararac); 1308 kg AU SP25 hatch (carsales); the BP (2019-) row above covers the other MoreTorque Mazda3"),
    V("Hyundai Sonata", "DN8 2020 (MoreTorque HYUNDAI_SONATA_2020; commaSteeringControl)", "HYUNDAI_SONATA", "ICE", "MoreTorque; commaSteeringControl",
      "non-hybrid fingerprint per CANDIDATES.md"),
    V("Lexus RX", "AL20 2016 (MoreTorque LEXUS_RX_2016; held-out model of the headline split)", "LEXUS_RX", "ICE", "MoreTorque; commaCarSegments",
      "platform covers RX 350 (ICE) and RX 450h (HEV); the MoreTorque fingerprint is the non-hybrid variant per CANDIDATES.md"),
    W("Omoda E5", "2024- (MoreTorque OMODA_E5, the only BEV there)", 2610, 1550, 1550, 1776, "BEV",
      "https://www.carexpert.co.nz/omoda-jaecoo/omoda-e5/2025/exterior-and-dimensions", "medium", "MoreTorque",
      "Australian Omoda E5 specification sheet (Chery AU dealer PDF) and carexpert: wheelbase 2610, kerb 1776 kg (NZ 2024 listing 1710 kg); "
      "evkx lists 2630 (-> 4.08); tracks 1550 / 1550 from carexpert / evkx; SMALL either way"),
    # ---------------------------------------------------------------- commaSteeringControl archives named in CANDIDATES 4.2
    V("Kia EV6", "CV 2022 (commaSteeringControl KIA_EV6_2022)", "KIA_EV6", "BEV", "commaSteeringControl; commaCarSegments"),
    V("Chevrolet Bolt EUV", "2022 (commaSteeringControl CHEVROLET_BOLT_EUV_2022)", "CHEVROLET_BOLT_EUV", "BEV", "commaSteeringControl; commaCarSegments",
      "platform also covers the Bolt EV 2022-23 hatch (2600/1501/1501 -> 3.90)"),
    V("Kia Niro EV", "DE 1st gen 2019-2022 (commaSteeringControl KIA_NIRO_EV_2020)", "KIA_NIRO_EV", "BEV", "commaSteeringControl; commaCarSegments"),
    V("Nissan Leaf", "ZE1 2018-2023 (commaSteeringControl NISSAN_LEAF_2018 and _Instrument_Cluster, one car)", "NISSAN_LEAF", "BEV",
      "commaSteeringControl; commaCarSegments", "held-out model of the headline split"),
    V("Hyundai Kona Electric", "OS 2018-2021 (commaSteeringControl HYUNDAI_KONA_ELECTRIC_2019)", "HYUNDAI_KONA_EV", "BEV",
      "commaSteeringControl; commaCarSegments", "held-out model of the headline split"),
    V("Genesis GV60", "2023 (commaSteeringControl GENESIS_GV60_ELECTRIC_1ST_GEN)", "GENESIS_GV60_EV_1ST_GEN", "BEV",
      "commaSteeringControl; commaCarSegments", "held-out model of the headline split"),
    V("Hyundai Ioniq Electric", "AE 2019 Limited (commaSteeringControl HYUNDAI_IONIQ_ELECTRIC_LIMITED_2019)", "HYUNDAI_IONIQ_EV_LTD", "BEV",
      "commaSteeringControl; commaCarSegments", "held-out model of the headline split"),
    V("Honda Accord", "10th gen 2018-2022 (commaSteeringControl HONDA_ACCORD_2018)", "HONDA_ACCORD", "ICE", "commaSteeringControl; commaCarSegments",
      "platform covers the Accord Hybrid as well; the archive name carries no hybrid tag"),
    V("Toyota Camry", "XV70 2018-2020 (commaSteeringControl TOYOTA_CAMRY)", "TOYOTA_CAMRY", "ICE", "commaSteeringControl; commaCarSegments",
      "0.038 above the MID cut; the hybrid has its own archive (HEV, excluded)"),
    V("RAM 1500", "DT 5th gen 2019- (commaSteeringControl RAM_1500_5TH_GEN)", "RAM_1500_5TH_GEN", "ICE", "commaSteeringControl; commaCarSegments",
      f"eTorque 48 V {MHEV}"),
    V("Honda Odyssey", "RL6 5th gen 2018-2020 (commaSteeringControl)", "HONDA_ODYSSEY", "ICE", "commaSteeringControl; commaCarSegments"),
    V("Chevrolet Silverado 1500", "T1 2020-2021 (commaSteeringControl)", "CHEVROLET_SILVERADO", "ICE", "commaSteeringControl; commaCarSegments"),
    V("Toyota Sienna", "XL30 3rd gen 2018-2020 (commaSteeringControl TOYOTA_SIENNA)", "TOYOTA_SIENNA", "ICE", "commaSteeringControl; commaCarSegments",
      "the 4th gen (2021-, hybrid only) is TOYOTA_SIENNA_4TH_GEN 3061/1720/1740 -> 5.30 LARGE, HEV"),
    # ---------------------------------------------------------------- tier 2: commaCarSegments keys absent from the lab table after MIGRATION
    V("Lexus NX", "AZ10 1st gen 2015-2021 (commaCarSegments LEXUS_NX = NX 300 2018-19, 378 segments)", "LEXUS_NX_TSS2", "ICE",
      "commaCarSegments (tier 2)", "lab row LEXUS_NX_TSS2 documents the 2020-21 facelift of the same body; the hybrid has its own opendbc platform (LEXUS_NXH)"),
    W("Skoda Octavia", "III 5E 2013-2020 (commaCarSegments SKODA_OCTAVIA_MK3, 322 segments)", 2686, 1549, 1540, 1263, "ICE",
      "https://www.auto-data.net/en/skoda-octavia-iii-facelift-2017-1.6-mpi-110hp-automatic-35793", "medium", "commaCarSegments (tier 2)",
      "tracks 1529-1551 / 1504-1520 by wheel size on other variants; Combi wheelbase 2680; kerb 1180-1300 kg by engine"),
    W("Genesis G80", "DH 1st gen 2017-2020 (commaCarSegments GENESIS_G80, 311 segments)", 3010, 1628, 1659, 1903, "ICE",
      "https://www.carspecs.us/cars/2018/genesis/g80", "medium", "commaCarSegments (tier 2)",
      "118.5 in, track 64.1 / 65.3 in (JD Power 63.8 front), curb 4195 lb (3.8 RWD); the lab table GENESIS_G80_2ND_GEN carries the same numbers; "
      "held-out model of the headline split", vtt_platform="GENESIS_G80_2ND_GEN"),
    W("Toyota Mirai", "JPD10 1st gen 2015-2020 (commaCarSegments TOYOTA_MIRAI, 299 segments)", 2780, 1535, 1545, 1848, "FCEV",
      "https://cars-data.com/en/toyota/mirai/2017/fcv-executive--92716/specs", "medium", "commaCarSegments (tier 2)",
      "FCEV: outside the BEV-vs-ICE task; US curb 4075 lb; the 2021+ Mirai (2920 mm) is a different car"),
    V("Toyota Avalon", "XX50 2019-2022 (commaCarSegments TOYOTA_AVALON_TSS2 and the legacy key 'TOYOTA AVALON 2022', 254 segments)",
      "TOYOTA_AVALON_2019", "unknown", "commaCarSegments (tier 2)",
      "platform covers Avalon and Avalon Hybrid 2019-22 (MIXED): the powertrain needs the lab Toyota ECU rule (hybrid ECU -> HEV, transmission ECU -> ICE)"),
    W("SEAT Ateca", "KH 2016- (commaCarSegments SEAT_ATECA_MK1, 153 segments)", 2638, 1576, 1541, 1274, "ICE",
      "https://mundoseat.seat.com/mediacenter_netstor/seat-media-center/global_site/documents/Ateca/en/Technical_Specifications_SEAT_ATECA_EN.pdf",
      "high", "commaCarSegments (tier 2)", "SEAT technical specifications: wheelbase 2638 (2630 4Drive), tracks 1576 / 1541; kerb 1249-1415 kg by drive"),
    W("Volkswagen T-Roc", "A11 2017- (commaCarSegments VOLKSWAGEN_TROC_MK1, 150 segments)", 2590, 1542, 1537, 1330, "ICE",
      "https://www.auto-data.net/en/volkswagen-t-roc-i-1.5-tsi-150hp-act-36112", "medium", "commaCarSegments (tier 2)",
      "tracks listed as 1538-1546 / 1533-1541 (mid-points used; cars-data 1537 / 1546); kerb 1250-1350 kg"),
    W("Acura RDX", "TB3/TB4 2nd gen 2013-2018 (commaCarSegments ACURA_RDX = RDX 2016-18, 147 segments)", 2685, 1604, 1610, 1695, "ICE",
      "https://www.acuracertified.com/-/media/Acura-Certified-Images/Brochures/2016_RDX_Factsheet.ashx", "high", "commaCarSegments (tier 2)",
      "Acura 2016 RDX fact sheet: 105.7 in, track 63.1 / 63.4 in, curb 3737 lb FWD / 3902 lb AWD; the lab table ACURA_RDX_3G is the 2019+ generation "
      "(2750/1631/1643 -> 4.502 MID), a different car", vtt_platform="ACURA_RDX_3G"),
    V("Kia Stinger", "CK facelift 2022 (commaCarSegments KIA_STINGER_2022, 100 segments)", "KIA_STINGER", "ICE", "commaCarSegments (tier 2)",
      "the facelift keeps the 2018-20 dimensions; held-out model of the headline split"),
    V("Hyundai Ioniq Plug-in", "AE 2019 (commaCarSegments HYUNDAI_IONIQ_PHEV_2019, 100 segments)", "HYUNDAI_IONIQ_PHEV", "PHEV",
      "commaCarSegments (tier 2)", "PHEV: outside the BEV-vs-ICE task; lab row documents the 2020-22 facelift of the same body"),
    W("Hyundai Elantra GT / i30", "PD 2017-2020 (commaCarSegments HYUNDAI_ELANTRA_GT_I30, 100 segments)", 2650, 1552, 1562, 1316, "ICE",
      "https://www.velocityjournal.com/models/30114/configuration/69131", "medium", "commaCarSegments (tier 2)",
      "104.3 in, track 61.1 / 61.5 in, curb 2901 lb (base 2.0)"),
    W("Lexus CT 200h", "ZWA10 2011-2017 (commaCarSegments LEXUS_CTH, 65 segments)", 2600, 1535, 1530, 1420, "HEV",
      "https://cars-data.com/en/lexus/ct/2014/200h-hybrid--75896/specs", "medium", "commaCarSegments (tier 2)",
      "HEV: outside the BEV-vs-ICE task; US curb 3130 lb"),
    W("Hyundai Staria", "US4 2021- (commaCarSegments HYUNDAI_STARIA_4TH_GEN, 61 segments)", 3273, 1721, 1732, 2255, "ICE",
      "https://www.auto-data.net/en/hyundai-staria-2.2-crdi-177hp-awd-automatic-7-seat-45594", "medium", "commaCarSegments (tier 2)",
      "kerb 2142 kg (3.5 V6 2WD) to 2438 kg (2.2 CRDi AWD); a hybrid version joined in 2024"),
    V("Lexus RC", "XC10 2015-2017 (commaCarSegments LEXUS_RC, 53 segments)", "LEXUS_RC_2017", "ICE", "commaCarSegments (tier 2)"),
    W("Toyota Prius v", "ZVW40 2012-2017 (commaCarSegments TOYOTA_PRIUS_V, 53 segments)", 2780, 1540, 1545, 1515, "HEV",
      "https://www.carexpert.com.au/toyota/prius-v/2015/exterior-and-dimensions", "medium", "commaCarSegments (tier 2)",
      "HEV: outside the BEV-vs-ICE task; US curb 3340 lb (AU 1565 kg)"),
    W("Buick Regal Sportback", "2nd gen (Insignia B) 2018-2020 (commaCarSegments BUICK_REGAL = Regal Essence 2018, 50 segments)", 2830, 1596, 1599, 1770,
      "ICE", "https://carbuzz.com/cars/buick/regal-sportback/2018/specs-and-trims", "medium", "commaCarSegments (tier 2)",
      "111.4 in, track 62.8 / 63.0 in, curb 3902 lb (Essence FWD)"),
    W("GMC Acadia", "C1XX 2nd gen 2017-2023 (commaCarSegments GMC_ACADIA = Acadia 2018, 50 segments)", 2857, 1638, 1638, 1794, "ICE",
      "https://www.carspecs.us/cars/2018/gmc/acadia", "medium", "commaCarSegments (tier 2)", "112.5 in, track 64.5 / 64.5 in, curb 3956 lb (base)"),
    W("Honda Freed", "GB5-GB8 2016-2024 (commaCarSegments HONDA_FREED, 50 segments)", 2740, 1480, 1485, 1360, "unknown",
      "https://www.carsales.com.au/research/honda/freed/2021/g", "medium", "commaCarSegments (tier 2)",
      "JDM compact MPV sold as petrol and e:HEV hybrid; the fingerprint does not say which -> unknown; tread 1480 / 1485 per JDM catalogues, 1340-1360 kg"),
    W("Skoda Kamiq", "NW4 2019- (commaCarSegments SKODA_KAMIQ_MK1, 35 segments)", 2651, 1531, 1516, 1151, "ICE",
      "https://cars-data.com/en/skoda/kamiq/nw4/1-0-tsi-116hp-greentech-style-98871--98871/specs", "medium", "commaCarSegments (tier 2)",
      "tracks per thecarspec 1531 / 1516; kerb 1139-1334 kg by engine and gearbox"),
    W("Dodge Durango", "WD 2011- (commaCarSegments DODGE_DURANGO = Durango 2020-21, 15 segments)", 3043, 1623, 1628, 2123, "ICE",
      "https://www.auto123.com/en/specs/dodge/durango/dodge-durango-2020-sxt-ti-409561/", "medium", "commaCarSegments (tier 2)",
      "119.8 in, track 63.9 / 64.1 in, curb 4680 lb RWD / 4814 lb AWD"),
    W("Perodua Myvi", "D20N 3rd gen 2017- (commaCarSegments legacy key 'PERODUA MYVI', community-fork platform, 12 segments)", 2500, 1475, 1465, 1010, "ICE",
      "https://www.carbase.my/perodua/myvi/d20n/1.5-high-at-2018", "medium", "commaCarSegments (tier 2)",
      "fork-only key (dropped by the CANDIDATES.md selection rule); tracks 1475 / 1465 per auto-data (1485 / 1475 in a Singapore listing); 955-1015 kg"),
    W("Kia Sorento", "UM 3rd gen 2015-2020 (commaCarSegments KIA_SORENTO = Sorento 2018, 10 segments)", 2780, 1628, 1638, 1680, "ICE",
      "https://autobytel.com/kia/sorento/2018", "medium", "commaCarSegments (tier 2)",
      "109.4 in, track 64.1 / 64.5 in (carweek 64.3 / 64.7), curb 3704 lb (2.4 FWD) to 4101 lb; the lab table KIA_SORENTO_4TH_GEN is the 2021+ "
      "generation (4.639 MID); held-out model of the headline split", vtt_platform="KIA_SORENTO_4TH_GEN"),
    V("Cadillac Escalade ESV", "4th gen 2015-2020 (commaCarSegments CADILLAC_ESCALADE_ESV_2019, 8 segments)", "CADILLAC_ESCALADE_ESV", "ICE",
      "commaCarSegments (tier 2)", "lab row documents the 2016 ESV of the same generation"),
    W("Cadillac Escalade", "4th gen standard wheelbase 2015-2020 (commaCarSegments CADILLAC_ESCALADE, 1 segment)", 2946, 1745, 1742, 2540, "ICE",
      "https://www.cars.com/research/cadillac-escalade-2016/specs", "medium", "commaCarSegments (tier 2)",
      "116.0 in; same tracks as the ESV (68.7 / 68.6 in); curb about 5600 lb"),
]


def build() -> list[dict]:
    vtt = load_vtt()
    out = []
    for r in ROWS:
        if r["kind"] == "vtt":
            p = vtt[r["platform"]]
            wb, tf, tr = float(p["wheelbase_mm"]), float(p["track_front_mm"]), float(p["track_rear_mm"])
            mass = r["mass"] if r["mass"] is not None else (float(p["curb_mass_kg"]) if p["curb_mass_kg"] else "")
            u = clean_url(p["source_urls"].split())
            src = f"VTT table: {r['platform']}" + (f" ; {u}" if u else "")
            conf = VTT_CONF.get(p["spec_confidence"].strip(), "medium")
            vtt_platform, spec_source = r["platform"], "VTT table"
        else:
            wb, tf, tr, mass = float(r["wb"]), float(r["tf"]), float(r["tr"]), r["mass"]
            src, conf = r["url"], r["conf"]
            vtt_platform, spec_source = r["vtt_platform"], "web"
        assert r["pt"] in POWERTRAINS, r["pt"]
        fp = footprint(wb, tf, tr)
        bl = is_borderline(fp) or bool(r.get("alt_cross"))
        out.append({"model": r["model"], "generation_or_years": r["gen"], "wheelbase_mm": int(round(wb)),
                    "track_front_mm": int(round(tf)), "track_rear_mm": int(round(tr)), "footprint_m2": f"{fp:.3f}",
                    "size_class": size_class(fp), "borderline": "true" if bl else "false",
                    "curb_mass_kg": int(round(float(mass))) if mass != "" else "", "powertrain": r["pt"], "source_url": src,
                    "confidence": conf, "note": r["note"], "datasets": r["datasets"], "vtt_platform": vtt_platform,
                    "spec_source": spec_source, "platform_keys": platform_keys(r["gen"], vtt_platform)})
    return out


def main() -> None:
    check_thresholds()
    rows = build()
    keys = [(r["model"], r["generation_or_years"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate model/generation rows"
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    # summary
    from collections import Counter
    print(f"wrote {OUT} with {len(rows)} rows")
    print("size_class:", dict(Counter(r["size_class"] for r in rows)))
    print("powertrain:", dict(Counter(r["powertrain"] for r in rows)))
    print("spec_source:", dict(Counter(r["spec_source"] for r in rows)))
    print("confidence:", dict(Counter(r["confidence"] for r in rows)))
    print("\nborderline rows:")
    for r in rows:
        if r["borderline"] == "true":
            print(f"  {r['model']:28s} {r['footprint_m2']} {r['size_class']:5s} {r['generation_or_years'][:60]}")
    print("\nall rows:")
    for r in rows:
        print(f"  {r['model']:28s} wb={r['wheelbase_mm']} tf={r['track_front_mm']} tr={r['track_rear_mm']} fp={r['footprint_m2']} "
              f"{r['size_class']:5s} bl={r['borderline']:5s} m={r['curb_mass_kg']:>5} {r['powertrain']:7s} {r['spec_source']:9s} {r['confidence']}")


if __name__ == "__main__":
    main()
