"""S8: device-disjoint and platform-disjoint splits, balanced evaluation lists, class weights and the split summary.

Units: hours = eligible 60 s windows / 60; shares are fractions of a cell's hours; "cell" = size_class x powertrain
with 'NA' for an unknown label (20 cells, 12 of them fully labelled).

Design (plan section 3, "划分"):
- unit = group_id (identity cluster); its load per cell is (hours, drives, presence);
- split_device: lab groups -> train (always); the other groups are placed greedily by scarcity (groups touching the
  smallest cells first) into the feasible split that minimises the squared relative deviation of every cell's
  (hours, drives, presence) from the SPLIT_FRACS targets; H2 cap: a group enters validation / test only if its hours are
  <= EVAL_GROUP_CAP x the split target in every cell whose target is >= CAP_MIN_TARGET_H; then single-move and swap
  refinement (deterministic, seeded tie order);
- split_platform (headline): whole platforms are held out per fully labelled cell (candidates: >= PLATFORM_HOLDOUT_MIN_GROUPS
  groups, no lab device, not the cell's top platform) by a subset-sum search that reaches PLATFORM_HOLDOUT_FRAC of the
  cell's hours counting the spill-over of every platform held out for the other cells (cells are swept in decreasing size
  until no choice changes) and prefers brands with both BEV and ICE held out; cells with an unknown size class or
  powertrain hold nothing out; drives of groups that also drove a held-out platform are 'excluded'; validation is carved
  group-disjoint from the remaining pool (15 % of pool hours) with the same engine;
- supported cells (SUPPORTED_CELL) get asserted shares (EVAL_SHARE_TOL) and group caps, other cells are reported;
  low_support (LOW_SUPPORT) is computed on the headline test set;
- balanced evaluation lists per task x split: round-robin over platforms then groups, caps BALANCED_EVAL, per-class
  count = min over classes (fallback: relax the platform cap, then the group cap, recorded in splits.json);
- train_uniform: weight = (N / (K n_k))^CLASS_WEIGHT_POWER capped at CLASS_WEIGHT_CAP;
- build/index/split_summary.csv columns: protocol (split_device | split_platform), task, class, split, hours, n_windows,
  n_drives, n_groups, n_vehicles, n_platforms (every name of the summary and of the splits/*.parquet lists must pass the
  release gate's flexible-table rule, checked before writing);
- INHOUSE_POLICY train_only needs private/inhouse_groups.csv with at least one lab group (else exit 2 unless --no-inhouse).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from vtt import config as C
from vtt import schema as S
from vtt.paths import P

SIZE_LEVELS = tuple(C.SIZE_CLASSES) + ("NA",)
PT_LEVELS = tuple(C.POWERTRAIN4) + ("NA",)
CELLS = tuple(f"{sz}|{pt}" for sz in SIZE_LEVELS for pt in PT_LEVELS)
CELL_INDEX = {c: i for i, c in enumerate(CELLS)}
DIMS = ("hours", "drives", "presence")
DEVICE_SPLITS = ("train", "validation", "test")
POOL_SPLITS = ("train", "validation")
PLATFORM_VALUES = ("train", "validation", "test", "excluded")
TASK_CLASSES = {"size": tuple(C.SIZE_CLASSES), "bev_ice": tuple(C.MAIN_POWERTRAIN_TASK)}
SPLITS_VERSION = "splits_v1"
WINDOW_H = C.WINDOW_SAMPLES / C.NOMINAL_HZ / 3600.0        # hours per eligible window (60 s)
SUMMARY_COLUMNS = ("protocol", "task", "class", "split", "hours", "n_windows", "n_drives", "n_groups", "n_vehicles", "n_platforms")
BALANCED_COLUMNS = ("window_id", "class", "group_id", "platform")
TRAIN_UNIFORM_COLUMNS = ("window_id", "weight")
HOLDOUT_MAX_PASSES = 6


class LockError(RuntimeError):
    """Size-class thresholds are not locked (config / docs/SIZE_CLASSES.md mismatch)."""


@dataclass
class SplitConfig:
    seed: int = C.SPLIT_SEED
    fracs: dict = field(default_factory=lambda: dict(C.SPLIT_FRACS))
    eval_group_cap: float = C.EVAL_GROUP_CAP
    cap_min_target_h: float = 5.0
    min_eval_groups: int = C.SUPPORTED_CELL["min_eval_groups"]
    min_eval_hours_frac: float = C.SUPPORTED_CELL["min_eval_hours_frac"]
    share_tol: float = C.EVAL_SHARE_TOL
    platform_holdout_frac: float = C.PLATFORM_HOLDOUT_FRAC
    platform_min_groups: int = C.PLATFORM_HOLDOUT_MIN_GROUPS
    pool_validation_frac: float = C.SPLIT_FRACS["validation"]
    balanced_max_group_share: float = C.BALANCED_EVAL["max_group_share"]
    balanced_max_platform_share: float = C.BALANCED_EVAL["max_platform_share"]
    balanced_min_per_class: int = 20
    low_support: dict = field(default_factory=lambda: dict(C.LOW_SUPPORT))
    class_weight_power: float = C.CLASS_WEIGHT_POWER
    class_weight_cap: float = C.CLASS_WEIGHT_CAP
    dim_weights: tuple = (1.0, 1.0, 1.0)
    refine_max_moves: int = 3000
    holdout_resolution: float = 0.005

    @property
    def device_eval_total(self) -> float:
        return float(sum(v for k, v in self.fracs.items() if k != "train"))

    def required_eval_hours_frac(self, eval_total: float) -> float:
        """min_eval_hours_frac is calibrated on the device split's eval share; scale it to another eval share."""
        return self.min_eval_hours_frac * eval_total / self.device_eval_total


# ----------------------------------------------------------------------------- paths and lock
def paths(data_root: Path | None = None) -> dict[str, Path]:
    root = Path(data_root) if data_root is not None else P.DATA
    rel = lambda p: root / p.relative_to(P.DATA)   # noqa: E731
    return {"drives": rel(P.DRIVES), "windows": rel(P.WINDOWS), "inhouse_groups": rel(P.INHOUSE_GROUPS),
            "pseudonym_map": rel(P.PSEUDONYM_MAP), "splits_dir": rel(P.SPLITS), "splits_json": rel(P.SPLITS_JSON),
            "split_summary": rel(P.SPLIT_SUMMARY), "private": rel(P.PRIVATE)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_locked(thresholds, doc_path: Path, expected_sha: str | None) -> tuple[bool, str]:
    """(ok, reason): thresholds set, docs/SIZE_CLASSES.md present and its sha256 equal to the configured one."""
    if thresholds is None or len(thresholds) != 2:
        return False, "config.FOOTPRINT_THRESHOLDS_M2 is not set (S3c not done)"
    if not expected_sha:
        return False, "config.SIZE_CLASSES_SHA256 is not set"
    doc_path = Path(doc_path)
    if not doc_path.exists():
        return False, "docs/SIZE_CLASSES.md is missing"
    got = sha256_file(doc_path)
    if got != expected_sha:
        return False, "sha256 of docs/SIZE_CLASSES.md differs from config.SIZE_CLASSES_SHA256"
    return True, "locked"


# ----------------------------------------------------------------------------- helpers
def _clean(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("", "nan", "none", "null") else s


def cell_of(size_class, powertrain) -> str:
    sz = _clean(size_class)
    pt = _clean(powertrain)
    sz = sz if sz in C.SIZE_CLASSES else "NA"
    pt = pt if pt in C.POWERTRAIN4 else "NA"
    return f"{sz}|{pt}"


def cell_parts(cell: str) -> tuple[str | None, str | None]:
    sz, pt = cell.split("|")
    return (None if sz == "NA" else sz), (None if pt == "NA" else pt)


def tie_hash(seed: int, *parts) -> int:
    return int(hashlib.sha256("|".join([str(seed), *map(str, parts)]).encode()).hexdigest()[:12], 16)


# ----------------------------------------------------------------------------- population
@dataclass
class Population:
    groups: np.ndarray            # (G,) group ids, sorted
    X: np.ndarray                 # (G, C, 3) hours / drives / presence
    is_lab: np.ndarray            # (G,) bool
    drives: pd.DataFrame          # drive_id, group_id, vehicle_id, platform, brand, cell, hours, g (group index), c (cell index)
    platforms: pd.DataFrame       # platform, brand, cell (dominant), hours, n_groups, has_lab, hours_by_cell (dict)

    @property
    def G(self) -> int:
        return len(self.groups)


def build_population(drives: pd.DataFrame, windows: pd.DataFrame, lab_groups: set[str]) -> Population:
    """Per-group loads per cell from the drives index and the eligible windows."""
    key = getattr(C, "HOLDOUT_KEY", "platform")          # hold-out unit: taxonomy model by default (user ruling), else platform
    if key not in drives.columns:
        raise ValueError(f"drives index lacks the hold-out key column {key!r}")
    d = drives[["drive_id", "group_id", "vehicle_id", key, "brand", "size_class", "powertrain"]].copy()
    if key != "platform":
        d = d.rename(columns={key: "platform"})          # 'platform' is the internal name of the hold-out unit throughout this module
    for c in ("drive_id", "group_id", "vehicle_id", "platform", "brand"):
        d[c] = d[c].map(_clean)
    if d.drive_id.duplicated().any():
        raise ValueError("drives index has duplicated drive_id")
    d["cell"] = [cell_of(s, p) for s, p in zip(d.size_class, d.powertrain)]
    w = windows[windows.eligible.astype(bool)]
    n_elig = w.groupby(w.drive_id.map(_clean)).size()
    d["hours"] = d.drive_id.map(n_elig).fillna(0).astype(float) * WINDOW_H
    groups = np.array(sorted(set(d.group_id)), dtype=object)
    gi = {g: i for i, g in enumerate(groups)}
    d["g"] = d.group_id.map(gi).astype(int)
    d["c"] = d.cell.map(CELL_INDEX).astype(int)
    X = np.zeros((len(groups), len(CELLS), 3))
    np.add.at(X[:, :, 0], (d.g.to_numpy(), d.c.to_numpy()), d.hours.to_numpy())
    np.add.at(X[:, :, 1], (d.g.to_numpy(), d.c.to_numpy()), 1.0)
    X[:, :, 2] = (X[:, :, 1] > 0).astype(float)
    is_lab = np.array([g in lab_groups for g in groups], dtype=bool)
    d["is_lab"] = is_lab[d.g.to_numpy()]
    rows = []
    for p, grp in d[d.platform != ""].groupby("platform", sort=True):
        hb = grp.groupby("cell").hours.sum().to_dict()
        dom = max(sorted(hb), key=lambda k: (hb[k], k))
        brands = grp.brand[grp.brand != ""]
        rows.append({"platform": p, "brand": brands.mode().iloc[0] if len(brands) else "", "cell": dom,
                     "hours": float(grp.hours.sum()), "n_groups": int(grp.group_id.nunique()),
                     "has_lab": bool(grp.is_lab.any()), "hours_by_cell": hb})
    plat = pd.DataFrame(rows, columns=["platform", "brand", "cell", "hours", "n_groups", "has_lab", "hours_by_cell"])
    return Population(groups=groups, X=X, is_lab=is_lab, drives=d, platforms=plat)


# ----------------------------------------------------------------------------- assignment engine
@dataclass
class Engine:
    """Greedy + refinement of groups into splits with per-cell targets.
    X: (G, C, D) loads; fracs: (S,); fixed: (G,) split index or -1; eval_splits: indices of the evaluation splits.
    Feasibility (H2): a group may enter an eval split only if its hours are <= eval_group_cap x the split's target in
    every cell whose target is >= cap_min_target_h and in every supported cell. Supported cells: >= min_eval_groups
    groups that fit under the cap of a target-size split and jointly hold >= the required share of the cell's hours."""
    X: np.ndarray
    fracs: np.ndarray
    fixed: np.ndarray
    eval_splits: tuple
    cfg: SplitConfig
    seed: int
    tag: str = ""

    def __post_init__(self):
        self.G, self.Cn, self.D = self.X.shape
        self.S = len(self.fracs)
        self.T = self.X.sum(0)                                        # (C, D)
        self.scale = np.where(self.T > 0, self.T, 1.0)
        self.w = (self.T > 0).astype(float) * np.asarray(self.cfg.dim_weights)[None, :]
        self.Xs = self.X / self.scale                                 # scaled loads (G, C, D)
        self.target = self.fracs[:, None, None] * self.T[None, :, :]  # (S, C, D)
        self.eval_total = float(sum(self.fracs[s] for s in self.eval_splits))
        self.fits_cap = self._fits_cap()                              # (G, C) group fits under the cap of every eval split's target
        self.global_ok = self._global_ok()                            # (G,) H2 in cells with target >= cap_min_target_h, not fixed
        self.supported = self._supported()                            # (C,)
        self.eval_ok = self._eval_ok()                                # (G, S)
        self.assign = np.full(self.G, -1, dtype=int)
        self.L = np.zeros((self.S, self.Cn, self.D))

    # -- feasibility
    def _fits_cap(self) -> np.ndarray:
        H = self.X[:, :, 0]
        ok = np.ones((self.G, self.Cn), dtype=bool)
        for s in self.eval_splits:
            lim = self.cfg.eval_group_cap * self.fracs[s] * self.T[:, 0]
            ok &= H <= lim[None, :] + 1e-12
        return ok

    def _global_ok(self) -> np.ndarray:
        big = np.zeros(self.Cn, dtype=bool)
        for s in self.eval_splits:
            big |= (self.fracs[s] * self.T[:, 0]) >= self.cfg.cap_min_target_h
        ok = self.fits_cap[:, big].all(1) if big.any() else np.ones(self.G, dtype=bool)
        return ok & (self.fixed < 0)

    def _supported(self) -> np.ndarray:
        H = self.X[:, :, 0]
        ev = self.fits_cap & self.global_ok[:, None] & (H > 0)
        n_ev = ev.sum(0)
        h_ev = (H * ev).sum(0)
        need = self.cfg.required_eval_hours_frac(self.eval_total) * self.T[:, 0]
        return (self.T[:, 0] > 0) & (n_ev >= self.cfg.min_eval_groups) & (h_ev >= need - 1e-9)

    def _eval_ok(self) -> np.ndarray:
        """(G, S) True when the group may enter split s."""
        ok = np.ones((self.G, self.S), dtype=bool)
        sup_ok = self.fits_cap[:, self.supported].all(1) if self.supported.any() else np.ones(self.G, dtype=bool)
        for s in self.eval_splits:
            ok[:, s] = self.global_ok & sup_ok
        for g in np.where(self.fixed >= 0)[0]:
            ok[g, :] = False
            ok[g, self.fixed[g]] = True
        return ok

    def evaluable(self) -> np.ndarray:
        """(G,) groups that may enter every eval split."""
        ev = self.fixed < 0
        for s in self.eval_splits:
            ev &= self.eval_ok[:, s]
        return ev

    def evaluable_in_cell(self) -> np.ndarray:
        """(G, C) evaluable groups with hours in the cell that fit under its cap (the support count)."""
        return self.fits_cap & self.evaluable()[:, None] & (self.X[:, :, 0] > 0)

    # -- objective
    def _A(self) -> np.ndarray:
        return (self.L - self.target) / self.scale[None, :, :]       # (S, C, D)

    def objective(self) -> float:
        A = self._A()
        return float((self.w[None, :, :] * A ** 2).sum())

    # -- greedy
    def order(self) -> np.ndarray:
        """Scarcity order: smallest touched cell total hours, then larger groups first, then seeded hash."""
        H = self.X[:, :, 0]
        Tc = self.T[:, 0]
        keys = []
        for g in range(self.G):
            touched = np.where(self.X[g, :, 1] > 0)[0]
            m = float(Tc[touched].min()) if len(touched) else float("inf")
            keys.append((m, -float(H[g].sum()), tie_hash(self.seed, self.tag, g)))
        return np.array(sorted(range(self.G), key=lambda g: keys[g]), dtype=int)

    def greedy(self) -> None:
        for g in np.where(self.fixed >= 0)[0]:
            self.assign[g] = self.fixed[g]
            self.L[self.fixed[g]] += self.X[g]
        for g in self.order():
            if self.assign[g] >= 0:
                continue
            A = self._A()
            x = self.Xs[g]
            dJ = (self.w[None, :, :] * ((A + x[None, :, :]) ** 2 - A ** 2)).sum((1, 2))
            dJ[~self.eval_ok[g]] = np.inf
            s = int(np.argmin(dJ))
            self.assign[g] = s
            self.L[s] += self.X[g]

    # -- refinement
    def _move_deltas(self) -> np.ndarray:
        """(G, S) change of the objective when a group moves to split s (inf where infeasible / same split)."""
        A = self._A()
        Xs = self.Xs.reshape(self.G, -1)
        Af = (A * self.w[None, :, :]).reshape(self.S, -1)
        Xw2 = ((self.w[None, :, :] * self.Xs ** 2).reshape(self.G, -1)).sum(1)
        remove = -2 * (Xs * Af[self.assign]).sum(1) + Xw2
        add = 2 * Xs @ Af.T + Xw2[:, None]
        dJ = remove[:, None] + add
        dJ[~self.eval_ok] = np.inf
        dJ[np.arange(self.G), self.assign] = np.inf
        return dJ

    def _apply(self, g: int, s: int) -> None:
        self.L[self.assign[g]] -= self.X[g]
        self.L[s] += self.X[g]
        self.assign[g] = s

    def _swap_deltas(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Objective change for every swap (i in a, j in b): 2 d.(A_b - A_a + d) with d = x_i - x_j (weighted)."""
        ia, ib = np.where(self.assign == a)[0], np.where(self.assign == b)[0]
        A = self._A().reshape(self.S, -1)
        W = self.w.reshape(-1)
        Xs = self.Xs.reshape(self.G, -1)
        v = (A[b] - A[a]) * W
        xa, xb = Xs[ia], Xs[ib]
        d_v = xa @ v
        d_vb = xb @ v
        gram = (xa * W) @ xb.T
        n2a = ((xa ** 2) * W).sum(1)
        n2b = ((xb ** 2) * W).sum(1)
        dd = n2a[:, None] + n2b[None, :] - 2 * gram
        dJ = 2 * (d_v[:, None] - d_vb[None, :]) + 2 * dd
        return dJ, ia, ib

    def refine(self) -> int:
        """Best improving single move per step, then the best improving swap, until none improves."""
        n_moves = 0
        while n_moves < self.cfg.refine_max_moves:
            dJ = self._move_deltas()
            k = int(np.argmin(dJ))
            g, s = divmod(k, self.S)
            if dJ[g, s] < -1e-9:
                self._apply(g, s)
                n_moves += 1
                continue
            if not self._best_swap():
                break
            n_moves += 1
        return n_moves

    def _best_swap(self) -> bool:
        best = (-1e-9, None)
        for a in range(self.S):
            for b in range(a + 1, self.S):
                dJ, ia, ib = self._swap_deltas(a, b)
                if dJ.size == 0:
                    continue
                feas = self.eval_ok[ia][:, b][:, None] & self.eval_ok[ib][:, a][None, :]
                dJ = np.where(feas, dJ, np.inf)
                k = int(np.argmin(dJ))
                i, j = divmod(k, dJ.shape[1])
                if dJ[i, j] < best[0]:
                    best = (float(dJ[i, j]), (int(ia[i]), int(ib[j]), a, b))
        if best[1] is None:
            return False
        gi, gj, a, b = best[1]
        self._apply(gi, b)
        self._apply(gj, a)
        return True

    # -- cap repair
    def shares(self) -> np.ndarray:
        """(S, C) hours share of each split per cell (0 where the cell is empty)."""
        Th = self.T[:, 0]
        return np.where(Th > 0, self.L[:, :, 0] / np.where(Th > 0, Th, 1.0), 0.0)

    def max_group_share(self) -> np.ndarray:
        """(S, C) largest single-group share of a split's hours in a cell."""
        H = self.X[:, :, 0]
        out = np.zeros((self.S, self.Cn))
        for s in range(self.S):
            idx = np.where(self.assign == s)[0]
            if len(idx) == 0:
                continue
            top = H[idx].max(0)
            L = self.L[s, :, 0]
            out[s] = np.where(L > 0, top / np.where(L > 0, L, 1.0), 0.0)
        return out

    def _violation_state(self) -> tuple[np.ndarray, np.ndarray]:
        """(band deviation beyond tolerance (S, C), cap excess (S, C)) on supported cells / eval splits, else 0."""
        sh = self.shares()
        mg = self.max_group_share()
        band = np.zeros((self.S, self.Cn))
        cap = np.zeros((self.S, self.Cn))
        for s in self.eval_splits:
            band[s] = np.maximum(0.0, np.abs(sh[s] - self.fracs[s]) - self.cfg.share_tol) * self.supported
            cap[s] = np.maximum(0.0, mg[s] - self.cfg.eval_group_cap) * self.supported
        return band, cap

    def repair(self, max_iter: int = 400) -> int:
        """Targeted exchanges that remove the asserted violations of supported cells (hours share outside the band,
        max group share above the cap): move a train group in, move a group out, or swap a group of the split with a
        train group; a candidate is accepted only when no violation grows anywhere and the targeted one shrinks;
        the smallest objective increase wins. Deterministic."""
        n = 0
        H = self.X[:, :, 0]
        train = 0
        for _ in range(max_iter):
            band0, cap0 = self._violation_state()
            if max(band0.max(), cap0.max()) <= 1e-9:
                break
            if band0.max() >= cap0.max():
                s, c = (int(v) for v in np.unravel_index(int(np.argmax(band0)), band0.shape))
                need = self.fracs[s] * self.T[c, 0] - self.L[s, c, 0]          # > 0: the split needs more hours in c
            else:
                s, c = (int(v) for v in np.unravel_index(int(np.argmax(cap0)), cap0.shape))
                need = 0.0
            in_s = np.where((self.assign == s) & (self.fixed < 0) & (H[:, c] > 0))[0]
            tr = np.where((self.assign == train) & (self.fixed < 0) & self.eval_ok[:, s] & (H[:, c] > 0))[0]
            cands: list[tuple[str, int, int]] = []
            if need > 0:
                cands += [("in", -1, int(g)) for g in tr]
                cands += [("swap", int(g), int(h)) for g in in_s for h in tr if H[h, c] > H[g, c]]
            elif need < 0:
                cands += [("out", int(g), -1) for g in in_s]
                cands += [("swap", int(g), int(h)) for g in in_s for h in tr if H[h, c] < H[g, c]]
            else:
                gstar = int(in_s[np.argmax(H[in_s, c])]) if len(in_s) else -1
                cands += [("in", -1, int(g)) for g in tr]
                if gstar >= 0:
                    cands += [("out", gstar, -1)]
                    cands += [("swap", gstar, int(h)) for h in tr if H[h, c] < H[gstar, c]]
            J0 = self.objective()
            target0 = band0[s, c] if need != 0 else cap0[s, c]
            best = None
            for kind, g_out, g_in in cands:
                if g_out >= 0:
                    self._apply(g_out, train)
                if g_in >= 0:
                    self._apply(g_in, s)
                band1, cap1 = self._violation_state()
                target1 = band1[s, c] if need != 0 else cap1[s, c]
                ok = (band1 <= band0 + 1e-12).all() and (cap1 <= cap0 + 1e-12).all() and target1 < target0 - 1e-12
                dJ = self.objective() - J0
                if g_in >= 0:
                    self._apply(g_in, train)
                if g_out >= 0:
                    self._apply(g_out, s)
                if ok and (best is None or dJ < best[0]):
                    best = (dJ, kind, g_out, g_in)
            if best is None:
                break
            _, kind, g_out, g_in = best
            if g_out >= 0:
                self._apply(g_out, train)
            if g_in >= 0:
                self._apply(g_in, s)
            n += 1
        return n


def run_engine(X: np.ndarray, fixed: np.ndarray, split_names: tuple, fracs: dict, cfg: SplitConfig, seed: int, tag: str) -> dict:
    """Greedy + refine + violation repair; returns the assignment and per-cell diagnostics."""
    fr = np.array([fracs[s] for s in split_names], dtype=float)
    eval_idx = tuple(i for i, s in enumerate(split_names) if s != "train")
    eng = Engine(X=X, fracs=fr, fixed=fixed, eval_splits=eval_idx, cfg=cfg, seed=seed, tag=tag)
    eng.greedy()
    n_moves = eng.refine()
    n_repair = eng.repair()
    return {"assign": np.array([split_names[s] for s in eng.assign], dtype=object), "assign_idx": eng.assign.copy(),
            "shares": eng.shares(), "max_group_share": eng.max_group_share(), "supported": eng.supported.copy(),
            "n_moves": n_moves, "n_repair": n_repair, "objective": eng.objective(), "evaluable": eng.evaluable(),
            "n_evaluable_cell": eng.evaluable_in_cell().sum(0), "T": eng.T.copy(), "L": eng.L.copy(),
            "split_names": tuple(split_names), "eval_idx": eval_idx}


# ----------------------------------------------------------------------------- platform hold-out
def _subset_search(items: list[tuple[str, float, int]], total: float, frac: float, tol: float,
                   resolution: float) -> tuple[list[str], bool]:
    """(subset, band_reachable): subset of (name, hours, bonus) whose hours share of total lies in [frac - tol, frac + tol]
    maximising (bonus sum, closeness to frac in 1 % steps, count); when no subset reaches the band, the reachable share
    closest to frac (the empty set included, so a lone oversized platform is not held out)."""
    if total <= 0 or not items:
        return [], False
    unit = total * resolution
    nb = int(round(1.0 / resolution)) + 1
    best: list = [None] * nb                                   # bin -> (bonus, count, subset tuple)
    best[0] = (0, 0, ())
    for name, hours, bonus in items:
        wgt = int(round(hours / unit))
        if wgt <= 0:
            wgt = 1
        for b in range(nb - 1, wgt - 1, -1):
            src = best[b - wgt]
            if src is None:
                continue
            cand = (src[0] + bonus, src[1] + 1, src[2] + (name,))
            if best[b] is None or cand[:2] > best[b][:2]:
                best[b] = cand
    hours_of = {n: h for n, h, _ in items}
    in_band, reachable = [], []
    for b in range(1, nb):
        if best[b] is None:
            continue
        subset = best[b][2]
        share = sum(hours_of[n] for n in subset) / total
        key = (best[b][0], -round(abs(share - frac) / 0.01), best[b][1], -abs(share - frac))   # pairing, closeness (1 % steps), count
        reachable.append((-abs(share - frac), best[b][0], best[b][1], subset))
        if frac - tol <= share <= frac + tol:
            in_band.append((key, subset))
    if in_band:
        return list(max(in_band, key=lambda t: t[0])[1]), True
    reachable.append((-abs(0.0 - frac), 0, 0, ()))
    return list(max(reachable, key=lambda t: t[:3])[3]), False


def cell_is_labelled(cell: str) -> bool:
    """True when both the size class and the powertrain of the cell are known (a task can evaluate it)."""
    sz, pt = cell_parts(cell)
    return sz is not None and pt is not None


def platform_holdout(pop: Population, cfg: SplitConfig, seed: int, max_passes: int = HOLDOUT_MAX_PASSES) -> tuple[set[str], dict]:
    """Held-out platforms per fully labelled cell with the brand-pairing preference. Returns (platforms, per-cell info).

    Candidates of a cell = platforms whose dominant cell it is, with >= platform_min_groups groups, no lab group and not the
    cell's top platform. A cell's choice is the subset of its candidates (subset-sum search over in-cell hours) whose hours
    plus the spill-over of every platform held out for the OTHER cells (their minority hours in this cell) reach
    PLATFORM_HOLDOUT_FRAC +- share_tol of the cell's hours. Because the spill-over depends on the other cells' choices,
    the cells are swept in decreasing size and the sweep is repeated until no cell changes its choice (at most
    ``max_passes`` sweeps), so every cell's final choice accounts for every held-out platform; ``band_reachable`` is
    judged on the final share. Cells with an unknown size class or powertrain never hold anything out (no task evaluates
    them; the spill-over they receive is reported). The info dict also carries ``_meta`` (passes, converged)."""
    plat = pop.platforms
    Tc = pop.X[:, :, 0].sum(0)
    cand = plat[(plat.n_groups >= cfg.platform_min_groups) & (~plat.has_lab) & (plat.platform != "")]
    brand_of = dict(zip(plat.platform, plat.brand))
    cell_of_plat = dict(zip(plat.platform, plat.cell))
    hb_of = dict(zip(plat.platform, plat.hours_by_cell))
    avail_class_by_brand: dict[str, set] = {}
    for p, b, c in zip(cand.platform, cand.brand, cand.cell):
        k = cell_parts(c)[1]
        if k in C.MAIN_POWERTRAIN_TASK and b:
            avail_class_by_brand.setdefault(b, set()).add(k)
    top_of_cell: dict[str, str] = {}
    for cell, grp in plat.groupby("cell"):
        top_of_cell[cell] = grp.sort_values(["hours", "platform"], ascending=[False, True]).platform.iloc[0]
    order = [ci for ci in sorted(range(len(CELLS)), key=lambda ci: (-Tc[ci], ci)) if Tc[ci] > 0]
    chosen_by_cell: dict[str, list[str]] = {}

    def spill_of(cell: str, held_other: set[str]) -> float:
        return float(sum(hb_of[p].get(cell, 0.0) for p in held_other))

    def choose(cell: str, held_other: set[str]) -> tuple[list[str], list[tuple[str, float, int]]]:
        total = float(Tc[CELL_INDEX[cell]])
        cc = cand[(cand.cell == cell) & (cand.platform != top_of_cell.get(cell, ""))]
        held_class_by_brand: dict[str, set] = {}
        for p in held_other:
            b, k = brand_of.get(p, ""), cell_parts(cell_of_plat[p])[1]
            if b and k in C.MAIN_POWERTRAIN_TASK:
                held_class_by_brand.setdefault(b, set()).add(k)
        k = cell_parts(cell)[1]
        items = []
        for p, b, hb in zip(cc.platform, cc.brand, cc.hours_by_cell):
            bonus = 0
            if k in C.MAIN_POWERTRAIN_TASK and b:
                other = (set(C.MAIN_POWERTRAIN_TASK) - {k}).pop()
                if other in held_class_by_brand.get(b, set()):
                    bonus = 2
                elif other in avail_class_by_brand.get(b, set()):
                    bonus = 1
            items.append((p, float(hb.get(cell, 0.0)), bonus))      # in-cell hours only
        items.sort()
        frac_left = cfg.platform_holdout_frac - spill_of(cell, held_other) / total
        if frac_left + cfg.share_tol > 0:
            chosen, _ = _subset_search(items, total, frac_left, cfg.share_tol, cfg.holdout_resolution)
        else:
            chosen = []
        return sorted(chosen), items

    passes, converged = 0, False
    for _ in range(max_passes):
        passes += 1
        changed = False
        for ci in order:
            cell = CELLS[ci]
            if not cell_is_labelled(cell):
                continue
            held_other = {p for c2, v in chosen_by_cell.items() if c2 != cell for p in v}
            chosen, _ = choose(cell, held_other)
            if chosen != chosen_by_cell.get(cell, []):
                changed = True
                chosen_by_cell[cell] = chosen
        if not changed:
            converged = True
            break
    held = {p for v in chosen_by_cell.values() for p in v}
    info: dict = {}
    for ci in order:
        cell = CELLS[ci]
        total = float(Tc[ci])
        chosen = chosen_by_cell.get(cell, [])
        held_other = held - set(chosen)
        spill = spill_of(cell, held_other)
        if cell_is_labelled(cell):
            _, items = choose(cell, held_other)
        else:
            items = []
        hours_held = float(sum(h for p, h, _ in items if p in chosen)) + spill
        share = hours_held / total
        entry = {"n_candidates": int(len(items)), "top_platform_excluded": top_of_cell.get(cell, ""), "heldout": sorted(chosen),
                 "band_reachable": bool(abs(share - cfg.platform_holdout_frac) <= cfg.share_tol + 1e-12),
                 "spill_hours": spill, "heldout_hours": hours_held, "heldout_share": share,
                 "candidate_hours_share": float(sum(h for _, h, _ in items)) / total,
                 "n_candidate_groups": int(sum(int(n) for p, n in zip(cand.platform, cand.n_groups) if p in {i[0] for i in items}))}
        if not cell_is_labelled(cell):
            entry["skipped"] = "unlabelled cell: no task evaluates it, nothing held out"
            entry["band_reachable"] = False
        info[cell] = entry
    info["_meta"] = {"passes": passes, "converged": converged}
    return held, info


# ----------------------------------------------------------------------------- balanced lists / weights
def balanced_list(win: pd.DataFrame, class_col: str, classes: tuple, cfg: SplitConfig, seed: int, tag: str) -> tuple[pd.DataFrame, dict]:
    """Round-robin over platforms then groups, caps on group / platform shares, equal count per class.
    win: eligible windows of one split with window_id, group_id, platform, drive_id, w_idx and class_col."""
    info: dict = {"fallbacks": [], "n_available": {}, "n_per_class": 0}
    pools = {}
    for k in classes:
        sub = win[win[class_col] == k].sort_values(["drive_id", "w_idx"]).reset_index(drop=True)
        info["n_available"][k] = int(len(sub))
        pools[k] = sub
    if any(len(v) == 0 for v in pools.values()):
        info["fallbacks"].append("empty_class")
        return pd.DataFrame(columns=["window_id", "class", "group_id", "platform"]), info

    def select(sub: pd.DataFrame, n: int, gcap: float | None, pcap: float | None) -> list[int]:
        """Row indices of up to n windows under the caps (None = no cap)."""
        if n <= 0:
            return []
        glim = max(1, math.floor(gcap * n)) if gcap is not None else n
        plim = max(1, math.floor(pcap * n)) if pcap is not None else n
        by_plat: dict[str, dict[str, list[int]]] = {}
        for i, (p, g) in enumerate(zip(sub.platform, sub.group_id)):
            by_plat.setdefault(p, {}).setdefault(g, []).append(i)
        p_order = sorted(by_plat, key=lambda p: (tie_hash(seed, tag, "p", p), p))
        g_order = {p: sorted(gs, key=lambda g: (tie_hash(seed, tag, "g", g), g)) for p, gs in by_plat.items()}
        p_cnt = dict.fromkeys(by_plat, 0)
        g_cnt = {p: dict.fromkeys(gs, 0) for p, gs in by_plat.items()}
        pos = {p: {g: 0 for g in gs} for p, gs in by_plat.items()}
        out: list[int] = []
        while len(out) < n:
            progressed = False
            for p in sorted(p_order, key=lambda p: (p_cnt[p], p_order.index(p))):
                if p_cnt[p] >= plim:
                    continue
                picked = False
                for g in sorted(g_order[p], key=lambda g: (g_cnt[p][g], g_order[p].index(g))):
                    if g_cnt[p][g] >= glim or pos[p][g] >= len(by_plat[p][g]):
                        continue
                    out.append(by_plat[p][g][pos[p][g]])
                    pos[p][g] += 1; g_cnt[p][g] += 1; p_cnt[p] += 1
                    picked = True
                    break
                if picked:
                    progressed = True
                    break
            if not progressed:
                break
        return out

    def feasible_max(sub: pd.DataFrame, gcap, pcap) -> int:
        lo, hi = 0, len(sub)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(select(sub, mid, gcap, pcap)) >= mid:
                lo = mid
            else:
                hi = mid - 1
        return lo

    gcap, pcap = cfg.balanced_max_group_share, cfg.balanced_max_platform_share
    for relax in ("none", "platform_cap_relaxed", "group_cap_relaxed"):
        if relax == "platform_cap_relaxed":
            pcap = None
            info["fallbacks"].append(relax)
        if relax == "group_cap_relaxed":
            gcap = None
            info["fallbacks"].append(relax)
        n = min(feasible_max(pools[k], gcap, pcap) for k in classes)
        if n >= cfg.balanced_min_per_class:
            break
    info["n_per_class"] = int(n)
    info["caps"] = {"max_group_share": gcap, "max_platform_share": pcap}
    parts = []
    for k in classes:
        idx = select(pools[k], n, gcap, pcap)
        sel = pools[k].iloc[idx]
        parts.append(pd.DataFrame({"window_id": sel.window_id.to_numpy(), "class": k, "group_id": sel.group_id.to_numpy(),
                                   "platform": sel.platform.to_numpy()}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["window_id", "class", "group_id", "platform"])
    return out.sort_values(["class", "window_id"]).reset_index(drop=True), info


def class_weights(win: pd.DataFrame, class_col: str, classes: tuple, cfg: SplitConfig) -> dict[str, float]:
    """(N / (K n_k))^power capped; classes absent from the training windows get the cap."""
    n = {k: int((win[class_col] == k).sum()) for k in classes}
    N, K = sum(n.values()), len(classes)
    out = {}
    for k in classes:
        w = (N / (K * n[k])) ** cfg.class_weight_power if n[k] > 0 and N > 0 else cfg.class_weight_cap
        out[k] = float(min(w, cfg.class_weight_cap))
    return out


# ----------------------------------------------------------------------------- the whole split
@dataclass
class SplitResult:
    drives: pd.DataFrame          # drive_id, group_id, vehicle_id, platform, brand, cell, hours, split_device, split_platform
    groups: dict                  # group_id -> {split_device, split_platform}
    cells: list                   # per-cell dicts (splits.json contract)
    heldout: list
    low_support_cells: list
    violations: list
    balanced: dict                # (scheme, task, split) -> DataFrame
    balanced_info: dict
    train_uniform: dict           # (scheme, task) -> DataFrame
    weights: dict                 # scheme -> task -> class -> w
    summary: pd.DataFrame
    meta: dict


def _cell_split_stats(d: pd.DataFrame, win_elig: pd.DataFrame, col: str, values: tuple) -> dict:
    out = {}
    nw = win_elig.groupby("drive_id").size()
    for v in values:
        sub = d[d[col] == v]
        out[v] = {"hours": float(sub.hours.sum()), "groups": int(sub.group_id.nunique()),
                  "platforms": int(sub.platform[sub.platform != ""].nunique()), "drives": int(len(sub)),
                  "windows": int(nw.reindex(sub.drive_id).fillna(0).sum())}
    return out


def build_splits(drives: pd.DataFrame, windows: pd.DataFrame, lab_groups: set[str], cfg: SplitConfig | None = None,
                 seed: int | None = None) -> SplitResult:
    cfg = cfg or SplitConfig()
    seed = cfg.seed if seed is None else int(seed)
    pop = build_population(drives, windows, lab_groups)
    d = pop.drives
    G = pop.G
    violations: list[str] = []

    # ---- split_device
    fixed = np.where(pop.is_lab, 0, -1)
    dev = run_engine(pop.X, fixed, DEVICE_SPLITS, cfg.fracs, cfg, seed, "device")
    d["split_device"] = dev["assign"][d.g.to_numpy()]
    if (d.loc[d.is_lab, "split_device"] != "train").any():
        violations.append("device: a lab group is not in train")

    # ---- split_platform
    heldout, hinfo = platform_holdout(pop, cfg, seed)
    gstar = set(d.loc[d.platform.isin(heldout), "group_id"])
    pool_mask = ~np.array([g in gstar for g in pop.groups])
    pool_idx = np.where(pool_mask)[0]
    pool_fr = {"train": 1.0 - cfg.pool_validation_frac, "validation": cfg.pool_validation_frac}
    pool = run_engine(pop.X[pool_idx], np.where(pop.is_lab[pool_idx], 0, -1), POOL_SPLITS, pool_fr, cfg, seed, "pool")
    gsplit_platform = {}
    for g in pop.groups:
        gsplit_platform[g] = "test" if g in gstar else None
    for i, gi in enumerate(pool_idx):
        gsplit_platform[pop.groups[gi]] = pool["assign"][i]
    sp = []
    for p, g in zip(d.platform, d.group_id):
        if p in heldout:
            sp.append("test")
        elif g in gstar:
            sp.append("excluded")
        else:
            sp.append(gsplit_platform[g])
    d["split_platform"] = sp
    if (d.loc[d.is_lab, "split_platform"].isin(("test",))).any():
        violations.append("platform: a lab group drives a held-out platform")
    if set(d.loc[d.split_platform.isin(("train", "validation")), "platform"]) & heldout:
        violations.append("platform: a held-out platform appears in train/validation")

    # ---- per-cell reporting + assertions
    win_elig = windows[windows.eligible.astype(bool)].copy()
    win_elig["drive_id"] = win_elig.drive_id.map(_clean)
    Tc = pop.X[:, :, 0].sum(0)
    cells = []
    low_support_cells = []
    dev_shares, dev_mgs = dev["shares"], dev["max_group_share"]
    pool_T = pool["T"][:, 0]
    for ci, cell in enumerate(CELLS):
        sz, pt = cell_parts(cell)
        sub = d[d.c == ci]
        if len(sub) == 0 and Tc[ci] == 0:
            continue
        dstats = _cell_split_stats(sub, win_elig, "split_device", DEVICE_SPLITS)
        pstats = _cell_split_stats(sub, win_elig, "split_platform", PLATFORM_VALUES)
        total_h = float(Tc[ci])
        test_h = pstats["test"]["hours"]
        test_share = test_h / total_h if total_h > 0 else 0.0
        pool_h = float(pool_T[ci])
        val_share_pool = pstats["validation"]["hours"] / pool_h if pool_h > 0 else 0.0
        # max group share of the platform test / validation sets
        mg_test = _max_group_share(sub[sub.split_platform == "test"])
        mg_val = _max_group_share(sub[sub.split_platform == "validation"])
        sup_dev = bool(dev["supported"][ci])
        sup_pool = bool(pool["supported"][ci])
        hi = hinfo.get(cell, {})
        sup_test = bool(total_h > 0 and hi.get("band_reachable", False) and hi.get("n_candidate_groups", 0) >= cfg.min_eval_groups
                        and hi.get("candidate_hours_share", 0.0) >= cfg.required_eval_hours_frac(cfg.platform_holdout_frac) - 1e-9)
        ls = cfg.low_support
        low = bool(test_h < ls["min_test_hours"] or pstats["test"]["groups"] < ls["min_groups"]
                   or pstats["test"]["platforms"] < ls["min_platforms"])
        low_dev = bool(dstats["test"]["hours"] < ls["min_test_hours"] or dstats["test"]["groups"] < ls["min_groups"]
                       or dstats["test"]["platforms"] < ls["min_platforms"])
        if low:
            low_support_cells.append(cell)
        # assertions
        if sup_dev:
            for s in ("validation", "test"):
                si = DEVICE_SPLITS.index(s)
                if abs(dev_shares[si, ci] - cfg.fracs[s]) > cfg.share_tol + 1e-9:
                    violations.append(f"device {cell} {s}: share {dev_shares[si, ci]:.3f} outside tolerance")
                if dev_mgs[si, ci] > cfg.eval_group_cap + 1e-9:
                    violations.append(f"device {cell} {s}: max group share {dev_mgs[si, ci]:.3f} > cap")
        if sup_test:
            if abs(test_share - cfg.platform_holdout_frac) > cfg.share_tol + 1e-9:
                violations.append(f"platform {cell} test: share {test_share:.3f} outside tolerance")
            if mg_test > cfg.eval_group_cap + 1e-9:
                violations.append(f"platform {cell} test: max group share {mg_test:.3f} > cap")
        if sup_pool:
            if abs(val_share_pool - cfg.pool_validation_frac) > cfg.share_tol + 1e-9:
                violations.append(f"platform {cell} validation: pool share {val_share_pool:.3f} outside tolerance")
            if mg_val > cfg.eval_group_cap + 1e-9:
                violations.append(f"platform {cell} validation: max group share {mg_val:.3f} > cap")
        cells.append({
            "cell": cell, "size_class": sz, "powertrain": pt, "hours": total_h, "groups": int(sub.group_id.nunique()),
            "platforms": int(sub.platform[sub.platform != ""].nunique()), "drives": int(len(sub)),
            "split": pstats, "supported": sup_test, "low_support": low, "max_group_share": {"validation": mg_val, "test": mg_test},
            "split_device": dstats, "split_platform": pstats,
            "supported_device": sup_dev, "supported_platform_test": sup_test, "supported_platform_validation": sup_pool,
            "low_support_device": low_dev,
            "shares_device": {s: float(dev_shares[i, ci]) for i, s in enumerate(DEVICE_SPLITS)},
            "max_group_share_device": {s: float(dev_mgs[i, ci]) for i, s in enumerate(DEVICE_SPLITS) if s != "train"},
            "test_share_platform": test_share, "validation_share_of_pool": val_share_pool, "pool_hours": pool_h,
            "holdout": hi,
        })

    # ---- group table
    groups = {g: {"split_device": dev["assign"][i], "split_platform": gsplit_platform[g]} for i, g in enumerate(pop.groups)}

    # ---- window-level frames for the lists
    w = windows.copy()
    for c in ("window_id", "drive_id", "group_id", "platform"):
        w[c] = w[c].map(_clean) if c in w else ""
    dmap = d.set_index("drive_id")
    w["split_device"] = w.drive_id.map(dmap.split_device)
    w["split_platform"] = w.drive_id.map(dmap.split_platform)
    w["platform"] = w.drive_id.map(dmap.platform)
    w["group_id"] = w.drive_id.map(dmap.group_id)
    w["cell"] = w.drive_id.map(dmap.cell)
    w["_size"] = w.cell.map(lambda c: cell_parts(c)[0] or "")
    w["_pt"] = w.cell.map(lambda c: cell_parts(c)[1] or "")
    we = w[w.eligible.astype(bool)]
    balanced, binfo, train_uniform, weights = {}, {}, {}, {}
    for scheme, col in (("platform", "split_platform"), ("device", "split_device")):
        weights[scheme] = {}
        for task, classes in TASK_CLASSES.items():
            ccol = "_size" if task == "size" else "_pt"
            elig = we[we[ccol].isin(classes)]
            for split in ("validation", "test"):
                lst, inf = balanced_list(elig[elig[col] == split], ccol, classes, cfg, seed, f"{scheme}|{task}|{split}")
                balanced[(scheme, task, split)] = lst
                binfo[f"{scheme}|{task}|{split}"] = inf
            tr = elig[elig[col] == "train"]
            cw = class_weights(tr, ccol, classes, cfg)
            weights[scheme][task] = cw
            train_uniform[(scheme, task)] = pd.DataFrame({"window_id": tr.window_id.to_numpy(),
                                                          "weight": tr[ccol].map(cw).astype(float).to_numpy()}).sort_values("window_id").reset_index(drop=True)
    summary = _summary(d, we)
    hmeta = hinfo.get("_meta", {})
    meta = {"n_groups": int(G), "n_lab_groups": int(pop.is_lab.sum()), "n_drives": int(len(d)), "n_heldout_platforms": len(heldout),
            "n_excluded_drives": int((d.split_platform == "excluded").sum()), "n_refine_moves_device": dev["n_moves"],
            "n_holdout_passes": int(hmeta.get("passes", 0)), "holdout_converged": bool(hmeta.get("converged", False)),
            "n_refine_moves_pool": pool["n_moves"], "n_repair_moves_device": dev["n_repair"], "n_repair_moves_pool": pool["n_repair"],
            "objective_device": dev["objective"], "objective_pool": pool["objective"],
            "n_supported_device": int(dev["supported"].sum()), "n_supported_platform_test": int(sum(c["supported"] for c in cells)),
            "n_supported_pool": int(pool["supported"].sum()), "n_low_support": len(low_support_cells), "n_violations": len(violations)}
    return SplitResult(drives=d, groups=groups, cells=cells, heldout=sorted(heldout), low_support_cells=low_support_cells,
                       violations=violations, balanced=balanced, balanced_info=binfo, train_uniform=train_uniform,
                       weights=weights, summary=summary, meta=meta)


def _max_group_share(sub: pd.DataFrame) -> float:
    h = sub.groupby("group_id").hours.sum()
    tot = float(h.sum())
    return float(h.max() / tot) if tot > 0 else 0.0


def _summary(d: pd.DataFrame, we: pd.DataFrame) -> pd.DataFrame:
    """SUMMARY_COLUMNS: protocol (= the split column, split_device | split_platform), task (all | cell | size | bev_ice),
    class, split, hours, n_windows, n_drives, n_groups, n_vehicles, n_platforms."""
    rows = []
    nw = we.groupby("drive_id").size()
    for col, values in (("split_device", DEVICE_SPLITS), ("split_platform", PLATFORM_VALUES)):
        for task in ("all", "cell", "size", "bev_ice"):
            if task == "all":
                keyed = [("all", d)]
            elif task == "cell":
                keyed = list(d.groupby("cell"))
            elif task == "size":
                keyed = [(k, g) for k, g in d.groupby("size_class") if _clean(k) in C.SIZE_CLASSES]
            else:
                keyed = [(k, g) for k, g in d.groupby("powertrain") if _clean(k) in C.MAIN_POWERTRAIN_TASK]
            for cls, sub in keyed:
                for v in values:
                    s = sub[sub[col] == v]
                    rows.append({"protocol": col, "task": task, "class": cls, "split": v, "hours": round(float(s.hours.sum()), 4),
                                 "n_windows": int(nw.reindex(s.drive_id).fillna(0).sum()), "n_drives": int(len(s)),
                                 "n_groups": int(s.group_id.nunique()), "n_vehicles": int(s.vehicle_id.nunique()),
                                 "n_platforms": int(s.platform[s.platform != ""].nunique())})
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def flex_column_problems(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Column names of the summary / list tables that the release gate's flexible-table rule would reject
    (not in vtt.gate.rules.SPLIT_TABLE_ALLOW or matching its deny pattern). Empty list = OK."""
    from vtt.gate import rules as R
    out = []
    for name, df in frames.items():
        bad = [c for c in df.columns if c not in R.SPLIT_TABLE_ALLOW or R.col_denied(c)]
        if bad:
            out.append(f"{name}: columns not allowed in a release table: {bad}")
    return out


# ----------------------------------------------------------------------------- outputs
def splits_json(res: SplitResult, cfg: SplitConfig, seed: int, sha_size_classes: str | None) -> dict:
    return {"version": SPLITS_VERSION, "seed": int(seed), "config": asdict(cfg), "config_version": C.CONFIG_VERSION,
            "schema_version": S.SCHEMA_VERSION, "headline": "split_platform", "cells": res.cells, "groups": res.groups,
            "heldout_platforms": res.heldout, "low_support_cells": res.low_support_cells,
            "class_weights": res.weights["platform"], "class_weights_device": res.weights["device"],
            "balanced_eval": res.balanced_info, "violations": res.violations, "counts": res.meta,
            "sha256_size_classes": sha_size_classes}


def _table_name(path: Path, default: str) -> str:
    import pyarrow.parquet as pq
    try:
        meta = pq.read_schema(path).metadata or {}
        return meta.get(b"vtt_table", default.encode()).decode()
    except Exception:
        return default


def write_list_parquet(df: pd.DataFrame, path: Path) -> Path:
    """A splits/*.parquet list with exactly the release metadata (vtt_table = file stem, vtt_schema_version), no pandas
    metadata and no index column, so build/splits is byte-compatible with what S9 publishes and the gate accepts it."""
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    table = table.replace_schema_metadata({b"vtt_table": path.stem.encode(), b"vtt_schema_version": S.SCHEMA_VERSION.encode()})
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return path


def output_frames(res: SplitResult) -> dict[str, pd.DataFrame]:
    """{relative file name: frame} of every flexible table S8 writes (lists + summary)."""
    out = {"index/split_summary.csv": res.summary}
    for (scheme, task, split), df in res.balanced.items():
        suffix = "" if scheme == "platform" else "_device"
        out[f"splits/eval_balanced_{task}_{split}{suffix}.parquet"] = df
    for (scheme, task), df in res.train_uniform.items():
        suffix = "" if scheme == "platform" else "_device"
        out[f"splits/train_uniform_{task}{suffix}.parquet"] = df
    return out


def write_outputs(res: SplitResult, cfg: SplitConfig, seed: int, p: dict[str, Path], drives: pd.DataFrame, windows: pd.DataFrame,
                  sha_size_classes: str | None) -> None:
    from vtt.private import write_json_atomic
    frames = output_frames(res)
    problems = flex_column_problems(frames)
    if problems:
        raise ValueError("; ".join(problems))
    p["splits_dir"].mkdir(parents=True, exist_ok=True)
    for rel, df in frames.items():
        if rel.startswith("splits/"):
            write_list_parquet(df, p["splits_dir"] / Path(rel).name)
    write_json_atomic(splits_json(res, cfg, seed, sha_size_classes), p["splits_json"])
    p["split_summary"].parent.mkdir(parents=True, exist_ok=True)
    res.summary.to_csv(p["split_summary"], index=False)
    # write back into the public indexes
    dm = res.drives.set_index("drive_id")
    low = {c["cell"]: c["low_support"] for c in res.cells}
    dr = drives.copy()
    key = dr.drive_id.map(_clean)
    dr["split_device"] = key.map(dm.split_device)
    dr["split_platform"] = key.map(dm.split_platform)
    dr["task_size_eligible"] = dr.size_class.map(_clean).isin(C.SIZE_CLASSES)
    dr["task_bev_ice_eligible"] = dr.powertrain.map(_clean).isin(C.MAIN_POWERTRAIN_TASK)
    dr["low_support_cell"] = key.map(dm.cell).map(low).fillna(True).astype(bool)
    S.write_public_parquet(dr, p["drives"], S.DRIVES_INDEX_COLUMNS, _table_name(p["drives"], "drives"))
    wn = windows.copy()
    wkey = wn.drive_id.map(_clean)
    wn["split_device"] = wkey.map(dm.split_device)
    wn["split_platform"] = wkey.map(dm.split_platform)
    wn["task_size_eligible"] = wkey.map(dr.set_index(key).task_size_eligible)
    wn["task_bev_ice_eligible"] = wkey.map(dr.set_index(key).task_bev_ice_eligible)
    wn["low_support_cell"] = wkey.map(dr.set_index(key).low_support_cell)
    S.write_public_parquet(wn, p["windows"], S.WINDOWS_INDEX_COLUMNS, _table_name(p["windows"], "windows"))


def lab_group_ids(p: dict[str, Path], require: bool = True) -> set[str]:
    """Lab groups from private/inhouse_groups.csv (group_id column) or via the pseudonym map (dongle_log join).
    With ``require`` (INHOUSE_POLICY train_only) a missing file raises FileNotFoundError and an empty group set raises
    ValueError, so the policy is never skipped silently on a wrong --data-root or before S4 ran."""
    from vtt.private import inhouse_groups
    if not p["inhouse_groups"].exists():
        if require:
            raise FileNotFoundError("private/inhouse_groups.csv is missing (run S4 first, or pass --no-inhouse to split without lab groups)")
        return set()
    ih = inhouse_groups(p["private"])
    out: set[str] = set()
    if "group_id" in ih.columns:
        out |= {g for g in ih.group_id.map(_clean) if g}
    if p["pseudonym_map"].exists() and len(ih):
        pm = pd.read_parquet(p["pseudonym_map"], columns=["dongle_log", "group_id"])
        out |= {g for g in pm.group_id[pm.dongle_log.isin(set(ih.dongle_log.map(_clean)))].map(_clean) if g}
    if require and not out:
        raise ValueError("private/inhouse_groups.csv lists no lab group (pass --no-inhouse to split without the train-only policy)")
    return out


def run(data_root: Path | None = None, size_classes_doc: Path | None = None, seed: int | None = None, dry_run: bool = False,
        cfg: SplitConfig | None = None, require_lock: bool = True, require_inhouse: bool = True) -> dict:
    """S8 end to end. Raises LockError when thresholds are not locked, FileNotFoundError / ValueError when the lab-group
    list is missing or empty (unless require_inhouse is False); returns counts (+ violations)."""
    cfg = cfg or SplitConfig()
    seed = cfg.seed if seed is None else int(seed)
    repo = Path(__file__).resolve().parents[1]
    doc = Path(size_classes_doc) if size_classes_doc is not None else repo / C.SIZE_CLASSES_DOC
    sha = None
    if require_lock:
        ok, reason = check_locked(C.FOOTPRINT_THRESHOLDS_M2, doc, C.SIZE_CLASSES_SHA256)
        if not ok:
            raise LockError(reason)
        sha = sha256_file(doc)
    p = paths(data_root)
    lab = lab_group_ids(p, require=require_inhouse)
    drives = pd.read_parquet(p["drives"])
    windows = pd.read_parquet(p["windows"])
    res = build_splits(drives, windows, lab, cfg, seed)
    counts = dict(res.meta)
    counts["violations"] = list(res.violations)
    counts["heldout_platforms"] = len(res.heldout)
    counts["dry_run"] = bool(dry_run)
    if res.violations:
        from vtt.private import write_json_atomic
        if not dry_run:
            p["splits_dir"].mkdir(parents=True, exist_ok=True)
            write_json_atomic(splits_json(res, cfg, seed, sha), p["splits_dir"] / "splits_FAILED.json")
        return counts
    if not dry_run:
        write_outputs(res, cfg, seed, p, drives, windows, sha)
    return counts
