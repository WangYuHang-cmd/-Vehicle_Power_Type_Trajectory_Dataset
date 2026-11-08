#!/usr/bin/env python3
"""Stage 5: the frozen final evaluation. Trains every entry of experiments/final_manifest.json on the full training split
with its pre-registered fixed epoch count and evaluates the test split exactly once per entry (an existing results file
is never re-run; a crash leaves no file, so a re-run after a crash is the only way a second evaluation can happen).

Manifest entries (JSON list under "entries"):
  {"kind": "window", "task": "size", "split_col": "split_platform", "model": "gru", "recipe_id": "S9",
   "kwargs": {run() keyword arguments of the recipe}, "fixed_epochs": 7, "seeds": [0, 1, 2]}
  {"kind": "mil", "split_col": "split_platform", "pool": "gated_attn", "recipe_id": "milB", "fixed_epochs": 20,
   "seeds": [0, 1, 2], "wemb_tags": ["split_platform__size__gru__s0", "split_platform__bev_ice__gru__s0"], "kwargs": {}}
  {"kind": "ensemble", "task": "bev_ice", "split_col": "split_platform", "name": "ens_gru_resnet", "how": "logodds",
   "members": ["split_platform__bev_ice__gru__S9", "split_platform__bev_ice__resnet__S9"], "seeds": [0, 1, 2]}
Window and MIL entries write the usual results JSON / npz through experiments.train.run and experiments.mil_train.run;
ensemble entries average the members' saved test probabilities (mean or log-odds mean over seeds and members) and
write a results JSON in the same format (model = name). Prints its own sha256 first so the hash can be recorded in
experiments/CAMPAIGN.md before the run; `--dry-run` only lists what would run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtt import config as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "experiments" / "final_manifest.json"
RESULTS = Path(C.DATA_ROOT) / "experiments" / "results"
PROTOCOLS = ("test_full", "test_balanced", "test_human_only")


def own_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def window_tag(e: dict, seed: int) -> str:
    rid = e.get("recipe_id", "")
    mid = f"{e['model']}__{rid}" if rid else e["model"]
    return f"{e['split_col']}__{e['task']}__{mid}__s{seed}"


def mil_tags(e: dict, seed: int) -> list[str]:
    model = "episode_mil" if e.get("pool", "gated_attn") == "gated_attn" else "episode_mil_meanlogit"
    model += e.get("variant", "")
    return [f"{e['split_col']}__{task}__{model}__s{seed}" for task in ("size", "bev_ice")]


def ensemble_tag(e: dict) -> str:
    return f"{e['split_col']}__{e['task']}__{e['name']}__s0"


def done(tag: str) -> bool:
    return (RESULTS / f"{tag}.json").exists()


# ----------------------------------------------------------------------------- ensembles from saved test probabilities
def ensemble_entry(e: dict, data, log=print) -> dict | None:
    from experiments.data import TASKS
    from experiments.train import drive_level, metrics, unit_protocols
    classes = TASKS[e["task"]]["classes"]
    members = []
    for m in e["members"]:
        for s in e.get("seeds", [0]):
            p = RESULTS / f"{m}__s{s}_probs.npz"
            if p.exists():
                members.append(p)
    if len(members) < 2:
        log(f"[{e['name']}] fewer than two member prediction files found; skipped")
        return None
    ref = np.load(members[0], allow_pickle=True)
    wid = ref["window_id"].astype(str)
    y = ref["y"].astype(np.int64)
    acc = np.zeros((len(wid), len(classes)), dtype=np.float64)
    for p in members:
        z = np.load(p, allow_pickle=True)
        if not np.array_equal(z["window_id"].astype(str), wid):
            raise SystemExit(f"member window order differs: {p.name}")
        pr = np.clip(z["probs"].astype(np.float64), 1e-4, 1.0)
        acc += np.log(pr) if e.get("how", "logodds") == "logodds" else pr
    probs = acc / len(members)
    if e.get("how", "logodds") == "logodds":
        probs = np.exp(probs - probs.max(1, keepdims=True))
    probs = probs / probs.sum(1, keepdims=True)
    # map the member windows (= test_full of the task) onto the window index
    pos = {w: i for i, w in enumerate(data.win["window_id"].to_numpy(object))}
    idx_all = np.array([pos[w] for w in wid], dtype=np.int64)
    res_prot, res_drive, keeps = {}, {}, {}
    for prot in PROTOCOLS:
        sel_idx = data.indices(e["task"], e["split_col"], "test", prot)
        want = set(int(i) for i in sel_idx)
        keep = np.array([int(i) in want for i in idx_all])
        keeps[prot] = keep
        res_prot[prot] = metrics(y[keep], probs[keep].argmax(1), classes)
        if prot != "test_balanced":
            res_drive[prot] = drive_level(data.win, idx_all[keep], probs[keep], y[keep], classes)
    kf, kh = keeps["test_full"], keeps["test_human_only"]
    units = unit_protocols(data, e["task"], e["split_col"], idx_all[kf], probs[kf], y[kf], idx_all[kh], probs[kh], y[kh], classes)
    tag = ensemble_tag(e)
    result = dict(tag=tag, task=e["task"], split_col=e["split_col"], model=e["name"], seed=0, n_params=0,
                  cfg=dict(kind="ensemble", how=e.get("how", "logodds"), members=[p.name for p in members]),
                  n_train=0, n_val=0, n_test=int(len(wid)), epochs_run=0, best_epoch=0, val_macro_f1=None, val_metrics={},
                  train_minutes=0.0, protocols=res_prot, drive_level=res_drive, unit_level=units, history=[])
    (RESULTS / f"{tag}.json").write_text(json.dumps(result, indent=1, default=float))
    np.savez_compressed(RESULTS / f"{tag}_probs.npz", window_id=wid, probs=probs.astype(np.float32), y=y)
    log(f"[{tag}] members={len(members)} test_full_f1={res_prot['test_full']['macro_f1']:.4f} drive_f1={res_drive['test_full']['macro_f1']:.4f}")
    return result


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma list of entry indices to run")
    ap.add_argument("--kinds", default="window,mil,ensemble", help="which kinds to run (ensembles need their members' files)")
    a = ap.parse_args(argv)
    print(f"eval_final.py sha256={own_sha256()}")
    man = json.loads(a.manifest.read_text())
    entries = man["entries"]
    only = {int(x) for x in a.only.split(",") if x.strip()} if a.only else None
    kinds = set(a.kinds.split(","))
    plan = []
    for i, e in enumerate(entries):
        if (only is not None and i not in only) or e["kind"] not in kinds:
            continue
        if e["kind"] == "window":
            for s in e.get("seeds", [0]):
                plan.append((i, e, s, window_tag(e, s), done(window_tag(e, s))))
        elif e["kind"] == "mil":
            for s in e.get("seeds", [0]):
                tags = mil_tags(e, s)
                plan.append((i, e, s, " + ".join(tags), all(done(t) for t in tags)))
        elif e["kind"] == "ensemble":
            plan.append((i, e, 0, ensemble_tag(e), done(ensemble_tag(e))))
    for i, e, s, tag, is_done in plan:
        print(f"[{i}] {e['kind']:8s} {tag}  {'done' if is_done else 'to run'}")
    if a.dry_run:
        return 0
    from experiments import train as T
    from experiments.data import WindowData
    data = None
    for i, e, s, tag, is_done in plan:
        if is_done:
            continue
        if e["kind"] == "window":
            data = data or WindowData()
            T.run(e["task"], e["split_col"], e["model"], s, data=data, fixed_epochs=e.get("fixed_epochs"), eval_test=True,
                  recipe_id=e.get("recipe_id", ""), log=lambda m: print(m, flush=True), **(e.get("kwargs") or {}))
        elif e["kind"] == "mil":
            from experiments import mil_train as MT
            tags = e.get("wemb_tags")
            MT.run(e["split_col"], s, pool=e.get("pool", "gated_attn"), val_mode=e.get("val_mode", "official"),
                   fixed_epochs=e.get("fixed_epochs"), eval_test=True, recipe_id=e.get("recipe_id", ""),
                   wemb_tags=",".join(tags) if isinstance(tags, list) else tags, wemb_allow_full=True,
                   variant=e.get("variant", ""), log=lambda m: print(m, flush=True), **(e.get("kwargs") or {}))
        elif e["kind"] == "ensemble":
            data = data or WindowData(in_ram=False)
            ensemble_entry(e, data, log=lambda m: print(m, flush=True))
    print("eval_final: finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
