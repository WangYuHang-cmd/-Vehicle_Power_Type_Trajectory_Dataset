"""Synthetic inputs for the S3a tests: a fake opendbc checkout, fake taxonomy tables and a fake data root.

Every identifier is generated here at run time (fake dongle ids 'f' * 10 + 6 hex digits, fake route ids); nothing is
copied from the real library.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from vtt import config as C

VALUES_TOYFAKE = '''
from opendbc.car import CarSpecs, PlatformConfig, Platforms, dbc_dict
from opendbc.car.common.conversions import Conversions as CV


class CAR(Platforms):
  FAKE_SEDAN = PlatformConfig(
    [FakeCarDocs("Fake Sedan 2017-19"), FakeCarDocs("Fake Sedan Hybrid 2017-19")],
    CarSpecs(mass=3505. * CV.LB_TO_KG, wheelbase=2.82, steerRatio=14.8, tireStiffnessFactor=0.7983),
    dbc_dict('fake_pt', 'fake_adas'),
  )
  FAKE_SEDAN_2020 = PlatformConfig(
    [FakeCarDocs("Fake Sedan 2020-22")],
    FAKE_SEDAN.specs,
    dbc_dict('fake_pt', 'fake_adas'),
  )
  FAKE_SEDAN_HEV_2020 = PlatformConfig(
    [FakeCarDocs("Fake Sedan Hybrid 2020-22")],
    FAKE_SEDAN.specs,
    dbc_dict('fake_pt', 'fake_adas'),
  )
  FAKE_SUV_EV = PlatformConfig(
    [FakeCarDocs("Fake SUV EV 2021-23")],
    specs=CarSpecs(mass=2100, wheelbase=2.9, steerRatio=13.0),
    dbc_dict=dbc_dict('fake_pt', None),
  )
  FAKE_SUV_EV_2ND_GEN = PlatformConfig(
    [FakeCarDocs("Fake SUV EV 2023-25")],
    CarSpecs(mass=2200, wheelbase=2.95, steerRatio=13.0),
    dbc_dict('fake_pt', None),
  )
'''

VALUES_GMFAKE = '''
from opendbc.car import CarSpecs, PlatformConfig, Platforms, dbc_dict


class GMCarSpecs(CarSpecs):
  tireStiffnessFactor: float = 0.444


class CAR(Platforms):
  FAKE_TRUCK = GMPlatformConfig(
    [GMCarDocs("Fake Truck 2019-21")],
    GMCarSpecs(mass=2500, wheelbase=3.745, steerRatio=16.3),
  )
  FAKE_TRUCK_IC = FAKE_TRUCK.override(car_docs=[])
  FAKE_TRUCK_LT = GMPlatformConfig(
    [GMCarDocs("Fake Truck LT 2022-24")],
    GMCarSpecs(mass=2550, wheelbase=3.745, steerRatio=16.3),
  )
  FAKE_TRUCK_HD = GMPlatformConfig(
    [GMCarDocs("Fake Truck HD 2022-24")],
    GMCarSpecs(mass=3200, wheelbase=4.0, steerRatio=16.3),
  )
'''

FINGERPRINTS = '''
from opendbc.car.toyfake.values import CAR as TOYFAKE
from opendbc.car.gmfake.values import CAR as GMFAKE

FW_VERSIONS = {}

MIGRATION = {
  "FAKE SEDAN 2017": TOYFAKE.FAKE_SEDAN,
  "FAKE TRUCK LEGACY": GMFAKE.FAKE_TRUCK,
}
'''

TAXONOMY_TABLES = '''
MODELS = {
 "Fake Sedan": ("Fake", "sedan", "D"), "Fake SUV": ("Fake", "suv", "SUV-C"), "Fake Truck": ("Fake", "pickup", "PU-F"),
 "Toyota Fake": ("Toyota", "suv", "SUV-C"), "Honda Accord": ("Honda", "sedan", "D"), "Volkswagen Golf": ("Volkswagen", "hatch", "C"),
 "Unknown": ("Unknown", "unknown", "UNK"),
}
DIRS = {
 "FAKE_SEDAN": ("Fake Sedan", "ICE", ""), "FAKE_SEDAN_2020": ("Fake Sedan", "ICE", ""), "FAKE_SEDAN_HEV_2020": ("Fake Sedan", "HEV", ""),
 "FAKE SEDAN 2017": ("Fake Sedan", "ICE", "legacy spelling"), "FAKE_SUV_EV": ("Fake SUV", "BEV", ""),
 "FAKE_TRUCK": ("Fake Truck", "ICE", "mild hybrid trims"), "TOYOTA_FAKE_TSS2": ("Toyota Fake", "MIXED", "gas and hybrid"),
 "HONDA_ACCORD": ("Honda Accord", "MIXED", ""), "VOLKSWAGEN_GOLF_MK7": ("Volkswagen Golf", "MIXED", ""),
 "Unknown": ("Unknown", "UNK", ""), "MOCK": ("Unknown", "UNK", ""),
}
ALIAS = {"FAKE_TRUCK_MMR": ("Fake Truck", "ICE")}
'''


def mkdtemp(prefix: str = "vtt_s3_") -> Path:
    base = C.DATA_ROOT / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def rmtree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)


def make_opendbc(root: Path) -> Path:
    car = root / "opendbc" / "car"
    (car / "toyfake").mkdir(parents=True, exist_ok=True)
    (car / "gmfake").mkdir(parents=True, exist_ok=True)
    (car / "toyfake" / "values.py").write_text(VALUES_TOYFAKE)
    (car / "gmfake" / "values.py").write_text(VALUES_GMFAKE)
    (car / "fingerprints.py").write_text(FINGERPRINTS)
    return root


def make_taxonomy_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "taxonomy_tables.py").write_text(TAXONOMY_TABLES)
    return d


def fake_dongle(i: int) -> str:
    return "f" * 10 + "%06x" % i


def fake_route(i: int) -> str:
    return f"2020-01-{1 + i % 28:02d}--00-{i // 60 % 60:02d}-{i % 60:02d}"


def fake_rel(car_dir: str, i: int) -> str:
    return f"{car_dir}/{fake_dongle(i)}/{fake_route(i)}"


# route scenarios: (car_dir, fingerprint, taxonomy row present?, link via other copy?, region, in_house, version, audit;
# fw = (model, powertrain) of a firmware-matched taxonomy row whose fingerprint is 'mock' and whose folder is Unknown)
SCENARIOS = [
    dict(car_dir="FAKE_SEDAN", fp="FAKE_SEDAN", tax=True, other_copy=False, region="US", in_house=False, version="0.9.7", audit=None),
    dict(car_dir="FAKE_SEDAN", fp="FAKE_SEDAN", tax=True, other_copy=True, region="US", in_house=False, version="0.8.16", audit=None),
    dict(car_dir="FAKE_SUV_EV", fp="FAKE_SUV_EV", tax=False, other_copy=False, region="Europe", in_house=False, version="2025.3.1", audit=None),
    dict(car_dir="Unknown", fp="TOTALLY_NEW_CAR", tax=False, other_copy=False, region="Other", in_house=False, version="0.10.1", audit=None),
    dict(car_dir="TOYOTA_FAKE_TSS2", fp="TOYOTA_FAKE_TSS2", tax=True, other_copy=False, region="Asia", in_house=False, version="0.9.6",
         audit=("eps;hybrid;fwdCamera", "unknown")),
    dict(car_dir="HONDA_ACCORD", fp="HONDA_ACCORD", tax=True, other_copy=False, region="US", in_house=False, version="0.11.0",
         audit=(json.dumps(["eps", "transmission"]), "automatic")),
    dict(car_dir="VOLKSWAGEN_GOLF_MK7", fp="VOLKSWAGEN_GOLF_MK7", tax=True, other_copy=False, region="Europe", in_house=False, version="0.9.7",
         audit=("eps;fwdRadar", "direct")),
    dict(car_dir="FAKE_TRUCK", fp=None, tax=True, other_copy=False, region="Canada", in_house=False, version="0.9.7", audit=None),
    dict(car_dir="FAKE_TRUCK", fp="FAKE_TRUCK_MMR", tax=True, other_copy=False, region="Canada", in_house=True, version="0.10.1", audit=None),
    dict(car_dir="FAKE_SEDAN", fp="mock", tax=True, other_copy=False, region=None, in_house=False, version="0.9.7", audit=None),
    dict(car_dir="Unknown", fp="mock", tax=True, other_copy=False, region="US", in_house=False, version="0.9.7", audit=None,
         fw=("Fake SUV", "BEV")),
]


def make_data_root(root: Path, taxonomy_dir: Path) -> dict:
    """private/routes_resolved.parquet + private/route_audit.parquet under root; taxonomy csv + master csv beside.
    Returns paths and the per-route expectations."""
    from vtt import labels as L

    tables = L.taxonomy_tables(taxonomy_dir)
    priv = root / "private"
    priv.mkdir(parents=True, exist_ok=True)
    os.chmod(priv, 0o700)
    (root / "build").mkdir(exist_ok=True)
    routes, tax_rows, master_rows, audit_rows = [], [], [], []
    for i, sc in enumerate(SCENARIOS):
        rel = fake_rel(sc["car_dir"], i)
        other = fake_rel(sc["car_dir"], 100 + i)
        uid = "%012x" % (0xABC000 + i)
        copies = [rel, other] if sc["other_copy"] else [rel]
        routes.append(dict(route_uid=uid, car_dir=sc["car_dir"], dongle_dir=fake_dongle(i), dongle_log=fake_dongle(i),
                           route_id=fake_route(i), identity_uid="%012x" % (0xDEF000 + i), vin_hash=None, rel_path=rel,
                           copies_rel=json.dumps(copies), n_copies=len(copies), n_segments=60 * (i + 1), n_qlog=60, n_rlog=0,
                           version=sc["version"], device_type="tici", car_fp=sc["fp"], opLong=False, radarUnavailable=False,
                           pcmCruise=True, probe_status="ok", dup_rule="none", is_long=False))
        tax_rel = other if sc["other_copy"] else rel
        if sc["tax"]:
            fp = sc["fp"]
            res = L.resolve(fp, tables) if fp is not None else None
            if sc.get("fw"):                     # mock fingerprint matched to a car through its firmware versions
                mdl, pt, how = sc["fw"][0], sc["fw"][1], "fw-match"
            elif res is None:                    # unreadable or mock fingerprint: the taxonomy build used the folder name
                mdl, pt, how = tables.DIRS[sc["car_dir"]][0], tables.DIRS[sc["car_dir"]][1], "directory"
            else:
                mdl, pt, how = res
            brand, body, size = L.model_attrs(mdl, tables)
            tax_rows.append(dict(rel_path=tax_rel, car_dir=sc["car_dir"], dongle=fake_dongle(i), fingerprint=fp, model=mdl, brand=brand,
                                 body=body, size=size, pt_platform=pt, powertrain=pt, source=how, dur_min=60 * (i + 1), miles=10.0,
                                 adas_min=5.0, group="g"))
        master_rows.append(dict(rel_path=tax_rel, region=sc["region"], country="Fakeland" if sc["region"] else None,
                                in_house=sc["in_house"], lat=12.345, lon=-67.89))
        if sc["audit"]:
            audit_rows.append(dict(route_uid=uid, ecus=sc["audit"][0], transmission=sc["audit"][1]))
    pd.DataFrame(routes).to_parquet(priv / "routes_resolved.parquet", index=False)
    pd.DataFrame(audit_rows).to_parquet(priv / "route_audit.parquet", index=False)
    tax_csv = root / "routes_taxonomy.csv"
    pd.DataFrame(tax_rows).to_csv(tax_csv, index=False)
    master_csv = root / "route_master.csv"
    pd.DataFrame(master_rows).to_csv(master_csv, index=False)
    return dict(root=root, private=priv, taxonomy_csv=tax_csv, master_csv=master_csv, routes=pd.DataFrame(routes),
                taxonomy=pd.DataFrame(tax_rows))
