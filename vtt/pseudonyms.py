"""S4: identity clusters (group), vehicle keys and random pseudonyms.

Everything in this module handles real identifiers (dongle ids, route ids, VIN hashes); only the public
``build/drive_names.parquet`` (route_uid, vehicle_id, drive_id, group_id) leaves the private root.

Rules (plan section 3, "匿名"):
- device key of a route = ``dongle_log`` (initData) else the folder re-key (misfiled dirs) else the folder name;
  a re-key whose value is not a 16-hex id (S1's "needs a manual entry" marker) is ignored and counted;
- group = union-find over device keys with four evidence rules:
  ``vin``        routes sharing a salted VIN hash (ignored when the hash is shared by > ``placeholder_max`` devices),
  ``route_id``   the same counter-style route id under two devices (same guard; date-style ids only within one car dir),
  ``explicit``   pairs listed in ``private/identity_merges.csv`` whose ``evidence`` is not in
                 ``config.MERGE_SKIP_EVIDENCE`` (a date-style id shared across car folders is a 1-s coincidence
                 candidate, the same case the route_id rule refuses; skipped pairs are counted per evidence),
  ``lab``        every in-house device (route_master in_house, extractor LAB / T0 lists, person-name folders) -> ONE
                 cluster; only 16-hex ids are accepted as lab devices;
- vehicle key = (device key, platform);
- pseudonyms = positions of a seeded random permutation of each namespace (vehicle_####, drive_######, group_####),
  handed out in sorted key order; the seed lives in ``private/release_seed.txt``;
- the map is append-only: an existing key never gets a new id and an id is never reused (``pseudonym_state.json`` keeps
  the counters and retired ids); clusters that merged or split since the last run need ``allow_regroup``; a vehicle id
  whose every route moved to another vehicle key (platform re-labelled) is retired, never reused.

Units: none (identifiers and counts only).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from vtt import config as C
from vtt.paths import P

HEX16 = re.compile(r"^[0-9a-f]{16}$")
_DATE_STYLE = re.compile(r"^\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}$")
_COUNTER_STYLE = re.compile(r"^[0-9a-f]{8}--[0-9a-f]{10}$")
TRUE_STRINGS = {"1", "1.0", "true", "True", "TRUE", "yes", "y"}
DEFAULT_PLACEHOLDER_MAX = 5           # fallback only; the extractor's IDENTITY_PLACEHOLDER_MAX_DONGLES is imported at run time

MAP_COLUMNS = ("route_uid", "vehicle_id", "drive_id", "group_id", "dongle_log", "platform", "route_id", "rel_path",
               "vin_hash", "assigned_release")
PUBLIC_COLUMNS = ("route_uid", "vehicle_id", "drive_id", "group_id")
NAMESPACES = {"vehicle": C.VEHICLE_FMT, "drive": C.DRIVE_FMT, "group": C.GROUP_FMT}
STATE_FILE = "pseudonym_state.json"
MERGE_REASONS = ("vin", "route_id", "explicit", "lab")


# ----------------------------------------------------------------------------- paths (re-rooted for --data-root)
def paths(data_root: Path | None = None) -> dict[str, Path]:
    """Every file S4 touches, under ``data_root`` (default config.DATA_ROOT); layout from vtt.paths.P."""
    root = Path(data_root) if data_root is not None else P.DATA
    rel = lambda p: root / p.relative_to(P.DATA)   # noqa: E731
    return {
        "private": rel(P.PRIVATE), "routes_resolved": rel(P.ROUTES_RESOLVED), "identity_merges": rel(P.IDENTITY_MERGES),
        "misfiled_dirs": rel(P.MISFILED_DIRS), "inhouse_groups": rel(P.INHOUSE_GROUPS), "pseudonym_map": rel(P.PSEUDONYM_MAP),
        "release_seed": rel(P.RELEASE_SEED), "state": rel(P.PRIVATE) / STATE_FILE,
        "routes_labels": rel(P.ROUTES_LABELS), "drive_names": rel(P.DRIVE_NAMES),
    }


# ----------------------------------------------------------------------------- small helpers
def clean(x) -> str:
    """None / NaN / 'nan' -> ''; everything else stripped str."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("", "nan", "none", "null") else s


def route_style(route_id: str) -> str:
    """'date' | 'counter' | 'other' (same convention as the extractor's rawlib.route_style)."""
    if _DATE_STYLE.fullmatch(route_id):
        return "date"
    if _COUNTER_STYLE.fullmatch(route_id):
        return "counter"
    return "other"


def device_key(dongle_log, dongle_dir, misfiled: dict[str, str] | None = None) -> str:
    """Device that recorded a route: initData's dongle id, else the folder re-key, else the folder name."""
    dl = clean(dongle_log)
    if dl:
        return dl
    dd = clean(dongle_dir)
    return (misfiled or {}).get(dd, dd)


def is_truthy(x) -> bool:
    return clean(x) in TRUE_STRINGS


def namespace_capacity(fmt: str) -> int:
    """'vehicle_{:04d}' -> 10**4 (the number of ids the format can hold)."""
    m = re.search(r"\{:0?(\d+)d\}", fmt)
    if not m:
        raise ValueError(f"cannot read the width of format {fmt!r}")
    return 10 ** int(m.group(1))


class UnionFind:
    """Union-find over string keys (path halving)."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True

    def members(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        for k in list(self.parent):
            out[self.find(k)].add(k)
        return dict(out)


# ----------------------------------------------------------------------------- private inputs
def load_merges(path: Path, skip_evidence=None) -> tuple[list[tuple[str, str]], dict]:
    """identity_merges.csv: header row, first two columns are device keys to merge; the optional ``evidence`` column
    (S1: counter_id | date_id_same_car | date_id_other_car) decides which pairs are used: a pair whose evidence is in
    ``skip_evidence`` (default config.MERGE_SKIP_EVIDENCE) is counted but not merged. Returns (pairs, counts) with
    counts = n_rows, n_used, n_skipped, by_evidence {evidence: rows}, skipped_evidence."""
    skip = set(C.MERGE_SKIP_EVIDENCE if skip_evidence is None else skip_evidence)
    counts = {"n_rows": 0, "n_used": 0, "n_skipped": 0, "by_evidence": {}, "skipped_evidence": sorted(skip)}
    if not Path(path).exists():
        return [], counts
    df = pd.read_csv(path, dtype=str).fillna("")
    if df.shape[1] < 2:
        return [], counts
    ev = df["evidence"].map(clean) if "evidence" in df.columns else pd.Series([""] * len(df), index=df.index)
    pairs: list[tuple[str, str]] = []
    for a, b, e in zip(df.iloc[:, 0], df.iloc[:, 1], ev):
        a, b = clean(a), clean(b)
        if not a or not b:
            continue
        counts["n_rows"] += 1
        key = e or "unspecified"
        counts["by_evidence"][key] = counts["by_evidence"].get(key, 0) + 1
        if e in skip:
            counts["n_skipped"] += 1
            continue
        pairs.append((a, b))
        counts["n_used"] += 1
    return pairs, counts


def filter_misfiled(raw: dict | None) -> tuple[dict[str, str], int]:
    """Keep the re-keys whose value is a 16-hex dongle id. S1 maps a non-hex folder that has no logged id anywhere to
    ITSELF (the 'needs a manual entry' marker) and its contract says consumers must ignore values that are not 16-hex.
    Returns (clean re-keys, number of ignored entries)."""
    out: dict[str, str] = {}
    ignored = 0
    for folder, dongle in (raw or {}).items():
        f, d = clean(folder), clean(dongle)
        if f and HEX16.match(d):
            out[f] = d
        else:
            ignored += 1
    return out, ignored


def load_misfiled(private_root: Path, use_tocf: bool = True) -> tuple[dict[str, str], int]:
    """(folder name -> dongle id re-keys, ignored non-hex markers) from the private JSON plus the extractor's static
    table when use_tocf (see filter_misfiled)."""
    if use_tocf:
        from vtt.private import misfiled_dirs
        return filter_misfiled(misfiled_dirs(Path(private_root)))
    p = Path(private_root) / P.MISFILED_DIRS.name
    return filter_misfiled(json.loads(p.read_text()) if p.exists() else {})


def tocf_lab_lists() -> dict[str, list[str]]:
    """{'tocf_lab': [...], 'tocf_t0_extra': [...]} imported from the extractor's config at run time (never copied)."""
    from vtt.tocf_bridge import tocf
    cfg = tocf("config")
    return {"tocf_lab": [clean(d) for d in getattr(cfg, "LAB_DONGLES", ())],
            "tocf_t0_extra": [clean(d) for d in getattr(cfg, "T0_EXTRA_IDENTITIES", ())]}


def tocf_placeholder_max() -> int:
    try:
        from vtt.tocf_bridge import tocf
        return int(tocf("config").IDENTITY_PLACEHOLDER_MAX_DONGLES)
    except Exception:
        return DEFAULT_PLACEHOLDER_MAX


def route_master_inhouse(route_master: pd.DataFrame | None, routes: pd.DataFrame) -> set[str]:
    """Device keys of routes flagged in_house in route_master (joined on any copy's rel_path; read-only)."""
    if route_master is None or "in_house" not in route_master or "rel_path" not in route_master:
        return set()
    flagged = {clean(p) for p, f in zip(route_master.rel_path, route_master.in_house) if is_truthy(f)}
    flagged.discard("")
    out: set[str] = set()
    for key, rel, copies in zip(routes["_device"], routes.get("rel_path", pd.Series([""] * len(routes))),
                                routes.get("copies_rel", pd.Series([""] * len(routes)))):
        paths_ = {clean(rel)}
        c = clean(copies)
        if c.startswith("["):
            try:
                paths_ |= {clean(x) for x in json.loads(c)}
            except ValueError:
                pass
        if paths_ & flagged:
            out.add(key)
    return out


def inhouse_devices(routes: pd.DataFrame, route_master: pd.DataFrame | None, misfiled: dict[str, str],
                    lab_lists: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Lab-owned devices: one row per device key with '+'-joined sources
    (route_master | tocf_lab | tocf_t0_extra | misfiled_name_dir). Person-name folders (non-hex keys of the re-key table)
    mark their logged device as in-house; hex-named re-keys are merge evidence only. Only 16-hex ids are accepted
    (a folder name or S1's self-mapped marker never becomes a lab device); the number of ignored values is in
    ``df.attrs['n_ignored_nonhex']``."""
    src: dict[str, set[str]] = defaultdict(set)
    ignored = 0

    def add(dev, name: str) -> None:
        nonlocal ignored
        d = clean(dev)
        if not d:
            return
        if not HEX16.match(d):
            ignored += 1
            return
        src[d].add(name)

    for d in route_master_inhouse(route_master, routes):
        add(d, "route_master")
    for name, devs in (lab_lists or {}).items():
        for d in devs:
            add(d, name)
    for folder, dongle in (misfiled or {}).items():
        if not HEX16.match(clean(folder)) and clean(dongle):
            add(dongle, "misfiled_name_dir")
    rows = [{"dongle_log": d, "source": "+".join(sorted(s))} for d, s in sorted(src.items())]
    df = pd.DataFrame(rows, columns=["dongle_log", "source"])
    df.attrs["n_ignored_nonhex"] = ignored
    return df


# ----------------------------------------------------------------------------- clusters
def prepare_routes(routes: pd.DataFrame, misfiled: dict[str, str]) -> pd.DataFrame:
    """Copy of routes_resolved with cleaned strings and the device key column ``_device``."""
    r = routes.copy()
    for c in ("route_uid", "car_dir", "dongle_dir", "dongle_log", "route_id", "vin_hash", "rel_path", "copies_rel"):
        r[c] = r[c].map(clean) if c in r else ""
    if (r.route_uid == "").any():
        raise ValueError("routes_resolved has rows without route_uid")
    if r.route_uid.duplicated().any():
        raise ValueError("routes_resolved has duplicated route_uid rows")
    r["_device"] = [device_key(dl, dd, misfiled) for dl, dd in zip(r.dongle_log, r.dongle_dir)]
    return r.reset_index(drop=True)


def build_groups(routes: pd.DataFrame, merges=(), inhouse: set[str] | None = None, misfiled: dict[str, str] | None = None,
                 placeholder_max: int = DEFAULT_PLACEHOLDER_MAX) -> tuple[pd.Series, dict]:
    """Identity clusters over the device keys of ``routes`` (prepared by prepare_routes).
    Returns (group key per route = smallest member device key, stats). The lab cluster contains every in-house device
    (also those without routes) so that lab membership is a single group."""
    inhouse = {clean(d) for d in (inhouse or set())} - {""}
    uf = UnionFind()
    n_merge = dict.fromkeys(MERGE_REASONS, 0)
    ignored = {"vin": 0, "route_id": 0}
    for d in routes["_device"]:
        uf.find(d)
    for folder, dongle in (misfiled or {}).items():          # a folder that still appears as a key joins its device
        if clean(folder) in uf.parent and clean(dongle):
            uf.union(clean(folder), clean(dongle))
    # vin rule
    sub = routes[routes.vin_hash != ""]
    for _, grp in sub.groupby("vin_hash", sort=True):
        devs = sorted(set(grp["_device"]))
        if len(devs) > placeholder_max:
            ignored["vin"] += 1
            continue
        for d in devs[1:]:
            n_merge["vin"] += uf.union(devs[0], d)
    # route_id rule
    sub = routes[routes.route_id != ""]
    if len(sub):
        key = [rid if route_style(rid) == "counter" else f"{car}/{rid}" for rid, car in zip(sub.route_id, sub.car_dir)]
        for _, grp in sub.groupby(np.array(key, dtype=object), sort=True):
            devs = sorted(set(grp["_device"]))
            if len(devs) > placeholder_max:
                ignored["route_id"] += 1
                continue
            for d in devs[1:]:
                n_merge["route_id"] += uf.union(devs[0], d)
    # explicit pairs
    for a, b in merges:
        n_merge["explicit"] += uf.union(a, b)
    # one lab cluster
    lab = sorted(inhouse)
    for d in lab:
        uf.find(d)
    for d in lab[1:]:
        n_merge["lab"] += uf.union(lab[0], d)
    members = uf.members()
    canon = {root: min(ms) for root, ms in members.items()}
    group_key = pd.Series([canon[uf.find(d)] for d in routes["_device"]], index=routes.index, name="group_key")
    lab_root = uf.find(lab[0]) if lab else None
    lab_key = canon[lab_root] if lab_root is not None else ""
    stats = {
        "n_routes": int(len(routes)), "n_devices": int(routes["_device"].nunique()),
        "n_devices_nonhex_key": int(sum(1 for d in set(routes["_device"]) if not HEX16.match(d))),
        "n_groups": int(group_key.nunique()), "n_merges": n_merge, "n_placeholders_ignored": ignored,
        "n_inhouse_devices": len(lab), "lab_group_key": lab_key,
        "n_lab_devices_in_cluster": int(len(members.get(lab_root, ()))) if lab_root is not None else 0,
        "n_lab_routes": int((group_key == lab_key).sum()) if lab else 0,
        "n_groups_multi_device": int(sum(1 for ms in members.values() if len(ms & set(routes["_device"])) > 1)),
    }
    return group_key, stats


# ----------------------------------------------------------------------------- seed, permutations, state
def read_release_seed(path: Path, create: bool = True) -> str:
    """Hex seed of the pseudonym permutations; created once with os.urandom (mode 600) when absent."""
    path = Path(path)
    if not path.exists():
        if not create:
            raise FileNotFoundError("release_seed.txt is missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(os.urandom(32).hex() + "\n")
    seed = path.read_text().strip()
    if len(seed) < 16:
        raise ValueError("release seed too short")
    return seed


def seed_fingerprint(seed_text: str) -> str:
    return hashlib.sha256(seed_text.encode()).hexdigest()[:16]


def permutation(seed_text: str, namespace: str, capacity: int) -> np.ndarray:
    """Deterministic random permutation of range(capacity) for one namespace."""
    h = hashlib.sha256(f"{seed_text}|{namespace}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    return rng.permutation(capacity)


def empty_state(seed_text: str) -> dict:
    return {"version": 1, "seed_sha256_16": seed_fingerprint(seed_text),
            "namespaces": {ns: {"n_assigned": 0, "retired": []} for ns in NAMESPACES}}


def load_state(path: Path, seed_text: str, existing_map: pd.DataFrame | None) -> dict:
    """pseudonym_state.json; rebuilt from the map when absent (only possible when no id was ever retired)."""
    path = Path(path)
    if path.exists():
        st = json.loads(path.read_text())
        if st.get("seed_sha256_16") != seed_fingerprint(seed_text):
            raise ValueError("release seed changed since the pseudonym map was created")
        return st
    st = empty_state(seed_text)
    if existing_map is not None and len(existing_map):
        for ns, col in (("vehicle", "vehicle_id"), ("drive", "drive_id"), ("group", "group_id")):
            st["namespaces"][ns]["n_assigned"] = int(existing_map[col].nunique())
    return st


class Namespace:
    """Hands out ids of one namespace: id(k) = fmt.format(perm[k]) for consecutive k; verifies the used set."""

    def __init__(self, name: str, fmt: str, seed_text: str, state: dict, used_ids: set[str]):
        self.name, self.fmt = name, fmt
        self.capacity = namespace_capacity(fmt)
        self.perm = permutation(seed_text, name, self.capacity)
        ns = state["namespaces"].setdefault(name, {"n_assigned": 0, "retired": []})
        self.n_assigned = int(ns["n_assigned"])
        self.retired = set(ns.get("retired", []))
        expected = {fmt.format(int(v)) for v in self.perm[: self.n_assigned]}
        if (used_ids | self.retired) != expected:
            raise ValueError(f"{name}: ids in the map/state do not match the first {self.n_assigned} positions of the permutation")

    def next_id(self) -> str:
        if self.n_assigned >= self.capacity:
            raise OverflowError(f"{self.name}: namespace exhausted")
        v = self.fmt.format(int(self.perm[self.n_assigned]))
        self.n_assigned += 1
        return v

    def retire(self, ids) -> None:
        self.retired |= set(ids)

    def dump(self) -> dict:
        return {"n_assigned": int(self.n_assigned), "retired": sorted(self.retired)}


# ----------------------------------------------------------------------------- assignment
def vehicle_key(device: str, platform: str) -> str:
    return f"{device}\t{clean(platform)}"


def _reconcile_groups(table: pd.DataFrame, old: pd.DataFrame | None) -> tuple[dict[str, str], dict[str, str], dict]:
    """Map new group keys to existing group ids (keep when a cluster still owns its old id).
    Returns (group_key -> old id or '', old_id -> owning group_key, counts). A cluster 'owns' an old id when it holds the
    most routes of that id (ties: smallest group key); every other overlap is a regroup (merge or split)."""
    counts = {"n_group_merged_ids": 0, "n_group_split_ids": 0}
    if old is None or not len(old):
        return {}, {}, counts
    old_gid = dict(zip(old.route_uid, old.group_id))
    pairs = defaultdict(int)                                  # (old_id, new key) -> routes
    for ru, gk in zip(table.route_uid, table.group_key):
        og = old_gid.get(ru)
        if og:
            pairs[(og, gk)] += 1
    by_old: dict[str, list] = defaultdict(list)
    for (og, gk), n in pairs.items():
        by_old[og].append((-n, gk))
    owner_of_old = {og: sorted(v)[0][1] for og, v in by_old.items()}
    counts["n_group_split_ids"] = sum(1 for v in by_old.values() if len(v) > 1)
    cands: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for og, gk in owner_of_old.items():
        cands[gk].append((-sum(n for (o, k), n in pairs.items() if o == og and k == gk), og))
    keep: dict[str, str] = {}
    for gk, lst in cands.items():
        lst.sort()
        keep[gk] = lst[0][1]
        if len(lst) > 1:
            counts["n_group_merged_ids"] += len(lst) - 1
    return keep, owner_of_old, counts


def assign_pseudonyms(table: pd.DataFrame, old: pd.DataFrame | None, state: dict, seed_text: str,
                      allow_regroup: bool = False, release: str = C.RELEASE_VERSION) -> tuple[pd.DataFrame, dict, dict]:
    """table: route_uid, dongle_log (device key), platform, group_key, route_id, rel_path, vin_hash (one row per route).
    old: the existing pseudonym map (or None). Returns (new map with MAP_COLUMNS, new state, counts).
    Raises AssertionError when an existing drive / vehicle-key / group assignment would change (unless allow_regroup
    for group changes)."""
    t = table.copy()
    t["vkey"] = [vehicle_key(d, p) for d, p in zip(t.dongle_log, t.platform)]
    old_rows = {} if old is None else {r.route_uid: r for r in old.itertuples(index=False)}
    used = {ns: set() for ns in NAMESPACES}
    vkey_to_id: dict[str, str] = {}
    if old is not None and len(old):
        used["vehicle"] = set(old.vehicle_id)
        used["drive"] = set(old.drive_id)
        used["group"] = set(old.group_id)
        for r in old.itertuples(index=False):
            vkey_to_id.setdefault(vehicle_key(r.dongle_log, r.platform), r.vehicle_id)
    spaces = {ns: Namespace(ns, fmt, seed_text, state, used[ns]) for ns, fmt in NAMESPACES.items()}
    counts = {"n_routes": int(len(t)), "n_new_routes": 0, "n_new_vehicles": 0, "n_new_groups": 0,
              "n_vehicle_rekeyed": 0, "n_group_changed_routes": 0}

    # vehicles: sorted new keys get the next permutation positions
    for vk in sorted(set(t.vkey) - set(vkey_to_id)):
        vkey_to_id[vk] = spaces["vehicle"].next_id()
        counts["n_new_vehicles"] += 1
    # groups
    keep, owner_of_old, gcounts = _reconcile_groups(t, old)
    counts.update(gcounts)
    gkey_to_id: dict[str, str] = {}
    for gk in sorted(set(t.group_key)):
        gid = keep.get(gk, "")
        if not gid:
            gid = spaces["group"].next_id()
            counts["n_new_groups"] += 1
        gkey_to_id[gk] = gid
    reused = set(gkey_to_id.values())
    retired = [og for og in (set(old.group_id) if old is not None and len(old) else set()) if og not in reused]
    # drives + rows
    rows = []
    for r in t.sort_values("route_uid").itertuples(index=False):
        o = old_rows.get(r.route_uid)
        if o is None:
            did = spaces["drive"].next_id()
            counts["n_new_routes"] += 1
            rel = release
        else:
            did, rel = o.drive_id, o.assigned_release
            if vehicle_key(o.dongle_log, o.platform) != r.vkey:
                counts["n_vehicle_rekeyed"] += 1
            if o.group_id != gkey_to_id[r.group_key]:
                counts["n_group_changed_routes"] += 1
        rows.append((r.route_uid, vkey_to_id[r.vkey], did, gkey_to_id[r.group_key], r.dongle_log, clean(r.platform),
                     clean(r.route_id), clean(r.rel_path), clean(r.vin_hash), rel))
    new = pd.DataFrame(rows, columns=list(MAP_COLUMNS))
    if counts["n_group_changed_routes"] and not allow_regroup:
        raise AssertionError(f"{counts['n_group_changed_routes']} routes would change group_id (clusters merged or split); "
                             "re-run with --allow-regroup to accept")
    if old is not None and len(old):
        missing = set(old.route_uid) - set(new.route_uid)
        if missing:                       # routes that vanished keep their rows (append-only map)
            new = pd.concat([new, old[old.route_uid.isin(missing)][list(MAP_COLUMNS)]], ignore_index=True)
            counts["n_routes_kept_from_old"] = len(missing)
    # a vehicle id that no row uses any more (every route of its key moved to another platform) is retired: it stays a
    # consumed permutation position, so the next run's namespace check still balances and the id is never handed out again
    retired_vehicle = sorted(set(old.vehicle_id) - set(new.vehicle_id)) if old is not None and len(old) else []
    spaces["vehicle"].retire(retired_vehicle)
    spaces["group"].retire(retired)
    new_state = dict(state)
    new_state["namespaces"] = {ns: sp.dump() for ns, sp in spaces.items()}
    new_state["seed_sha256_16"] = seed_fingerprint(seed_text)
    new_state["version"] = 1
    counts["n_group_ids_retired"] = len(retired)
    counts["n_vehicle_ids_retired"] = len(retired_vehicle)
    return new.sort_values("route_uid").reset_index(drop=True), new_state, counts


# ----------------------------------------------------------------------------- checks
def check_map(new: pd.DataFrame, old: pd.DataFrame | None = None, lab_devices: set[str] | None = None) -> list[str]:
    """Invariants of a pseudonym map (empty list = OK)."""
    problems = []
    fmts = {"vehicle_id": C.VEHICLE_FMT, "drive_id": C.DRIVE_FMT, "group_id": C.GROUP_FMT}
    for col, fmt in fmts.items():
        pat = re.compile("^" + re.escape(fmt.split("{")[0]) + r"\d{" + str(len(fmt.format(0)) - len(fmt.split("{")[0])) + "}$")
        bad = int((~new[col].astype(str).str.fullmatch(pat.pattern)).sum())
        if bad:
            problems.append(f"{col}: {bad} ids do not match the format")
    if new.route_uid.duplicated().any():
        problems.append("duplicate route_uid")
    if new.drive_id.duplicated().any():
        problems.append("duplicate drive_id")
    ids = [set(new.vehicle_id), set(new.drive_id), set(new.group_id)]
    if (ids[0] & ids[1]) or (ids[0] & ids[2]) or (ids[1] & ids[2]):
        problems.append("namespaces overlap")
    vk = pd.DataFrame({"vk": [vehicle_key(d, p) for d, p in zip(new.dongle_log, new.platform)], "vehicle_id": new.vehicle_id.to_numpy()})
    if (vk.groupby("vk").vehicle_id.nunique() > 1).any():
        problems.append("a vehicle key maps to several vehicle_ids")
    if (vk.groupby("vehicle_id").vk.nunique() > 1).any():
        problems.append("a vehicle_id maps to several vehicle keys")
    if (new.groupby("dongle_log").group_id.nunique() > 1).any():
        problems.append("a device maps to several group_ids")
    if lab_devices:
        lab_groups = set(new.group_id[new.dongle_log.isin(lab_devices)])
        if len(lab_groups) > 1:
            problems.append(f"lab devices span {len(lab_groups)} groups")
    if old is not None and len(old):
        m = old.merge(new, on="route_uid", suffixes=("_old", "_new"))
        if len(m) != len(old):
            problems.append("routes of the old map are missing from the new map")
        if (m.drive_id_old != m.drive_id_new).any():
            problems.append("drive_id changed for an existing route")
        ov = old.assign(vk=[vehicle_key(d, p) for d, p in zip(old.dongle_log, old.platform)])
        nv = new.assign(vk=[vehicle_key(d, p) for d, p in zip(new.dongle_log, new.platform)])
        j = ov[["vk", "vehicle_id"]].drop_duplicates().merge(nv[["vk", "vehicle_id"]].drop_duplicates(), on="vk")
        if (j.vehicle_id_x != j.vehicle_id_y).any():
            problems.append("vehicle_id changed for an existing vehicle key")
    return problems


def public_names(new: pd.DataFrame, route_uids=None) -> pd.DataFrame:
    """drive_names.parquet rows: route_uid, vehicle_id, drive_id, group_id (optionally only the given routes)."""
    df = new[list(PUBLIC_COLUMNS)]
    if route_uids is not None:
        df = df[df.route_uid.isin(set(route_uids))]
    return df.sort_values("route_uid").reset_index(drop=True)


# ----------------------------------------------------------------------------- end-to-end
def run(data_root: Path | None = None, route_master: Path | None = None, use_tocf: bool = True, allow_regroup: bool = False,
        allow_missing_labels: bool = False, dry_run: bool = False, require_route_master: bool = True) -> dict:
    """S4 end to end. Returns counts (no identifiers). Raises FileNotFoundError when the route_master table (in_house
    column) is absent unless ``require_route_master`` is False (the lab list then comes from the other sources only)."""
    from vtt.private import ensure_private_root, write_parquet_atomic, write_text_atomic

    p = paths(data_root)
    ensure_private_root(p["private"])
    routes = pd.read_parquet(p["routes_resolved"])
    misfiled, n_misfiled_ignored = load_misfiled(p["private"], use_tocf=use_tocf)
    r = prepare_routes(routes, misfiled)
    # platform per route
    platform = pd.Series([""] * len(r), index=r.index)
    if p["routes_labels"].exists():
        lab = pd.read_parquet(p["routes_labels"], columns=["route_uid", "platform"])
        platform = r.route_uid.map(dict(zip(lab.route_uid, lab.platform.map(clean)))).fillna("")
    elif not allow_missing_labels:
        raise FileNotFoundError("build/routes_labels.parquet missing (run S3a first or pass --allow-missing-labels)")
    else:
        platform = r.get("car_fp", pd.Series([""] * len(r))).map(clean)
    r["platform"] = platform.to_numpy()
    # in-house devices
    rm = None
    rm_path = Path(route_master) if route_master is not None else C.ROUTE_MASTER_CSV
    if rm_path.exists():
        rm = pd.read_csv(rm_path, dtype=str, usecols=["rel_path", "in_house"], low_memory=False)
    elif require_route_master:
        raise FileNotFoundError("route_master table (in_house column) not found; pass --route-master PATH or --no-route-master")
    lab_lists = tocf_lab_lists() if use_tocf else {}
    inhouse = inhouse_devices(r, rm, misfiled, lab_lists)
    n_inhouse_ignored = int(inhouse.attrs.get("n_ignored_nonhex", 0))
    lab_devices = set(inhouse.dongle_log)
    merges, merge_counts = load_merges(p["identity_merges"])
    placeholder_max = tocf_placeholder_max() if use_tocf else DEFAULT_PLACEHOLDER_MAX
    group_key, gstats = build_groups(r, merges, lab_devices, misfiled, placeholder_max)
    table = pd.DataFrame({"route_uid": r.route_uid, "dongle_log": r["_device"], "platform": r.platform, "group_key": group_key,
                          "route_id": r.route_id, "rel_path": r.rel_path, "vin_hash": r.vin_hash})
    old = pd.read_parquet(p["pseudonym_map"]) if p["pseudonym_map"].exists() else None
    seed_text = read_release_seed(p["release_seed"], create=not dry_run) if (p["release_seed"].exists() or not dry_run) \
        else os.urandom(32).hex()
    state = load_state(p["state"], seed_text, old)
    new, new_state, counts = assign_pseudonyms(table, old, state, seed_text, allow_regroup=allow_regroup)
    problems = check_map(new, old, lab_devices)
    inhouse = inhouse.assign(group_id=inhouse.dongle_log.map(dict(zip(new.dongle_log, new.group_id))).fillna(""))
    counts.update({"groups": gstats, "n_inhouse_devices": int(len(inhouse)),
                   "n_inhouse_devices_with_routes": int((inhouse.group_id != "").sum()),
                   "n_inhouse_nonhex_ignored": n_inhouse_ignored, "route_master_used": rm is not None,
                   "n_misfiled_rekeys": int(len(misfiled)), "n_misfiled_markers_ignored": int(n_misfiled_ignored),
                   "merges": merge_counts,
                   "n_platform_missing": int((r.platform == "").sum()), "n_vehicles": int(new.vehicle_id.nunique()),
                   "n_groups": int(new.group_id.nunique()), "problems": problems, "dry_run": bool(dry_run)})
    if "identity_uid" in r and (r.identity_uid.map(clean) != "").any():
        iu = r[r.identity_uid.map(clean) != ""]
        g_of = dict(zip(table.route_uid, table.group_key))
        counts["n_identity_uid_spanning_groups"] = int((iu.groupby("identity_uid").route_uid.apply(
            lambda s: len({g_of[x] for x in s})) > 1).sum())
    if problems or dry_run:
        return counts
    write_parquet_atomic(new, p["pseudonym_map"], private=True)
    write_text_atomic(json.dumps(new_state, indent=1, sort_keys=True) + "\n", p["state"], private=True)
    write_text_atomic(inhouse.to_csv(index=False), p["inhouse_groups"], private=True)
    write_parquet_atomic(public_names(new, r.route_uid), p["drive_names"])
    return counts
