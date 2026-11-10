#!/usr/bin/env python3
"""Channel-subset variants of the two final recipes, trained on our data and scored on our held-out car models.

The external datasets of the transfer test (CANDIDATES.md) lack whole channel groups, so the final models cannot be
applied to them as trained. This script trains, one process at a time, the final recipes of
experiments/final_manifest.json (GRU S9 for size, 11 fixed epochs; ResNet B5 for BEV vs ICE, 7 fixed epochs; same
kwargs) with an opt-in input mask (experiments.train.run(channels=...)): every channel outside the variant's list is set
to NaN, i.e. to the missing value 0, on train, validation and test, exactly as the scorer treats a channel an external
dataset cannot provide. Variants (recipe ids):
  V1 lateral + speed:    v_mps, a_mps2, standstill, das_enabled, steer_angle_deg, steer_rate_degps, yaw_rate_radps,
                         curv_1pm, lat_acc_mps2, steer_pressed               (commaSteeringControl-like inputs)
  V2 longitudinal-only:  v_mps, a_mps2, standstill, das_enabled, lead group   (ULTra-AV / OpenACC / UGA / microSIM)
  V3 speed-only:         v_mps, a_mps2, standstill                            (Yuan Tesla, VED, one-vehicle OpenACC)
Outputs follow the published layout with the tag prefix `transfer`: checkpoints <data-root>/experiments/ckpt/
transfer__<task>__<arch>__<Vk>__s<seed>.pt (cfg["channels"] = the list) and results <data-root>/experiments/results/
<tag>.json + <tag>_probs.npz (test_full window probabilities), which the transfer scorer loads by tag. Finished runs
(results JSON present) are skipped. `--render` writes experiments/transfer/VARIANTS.md from the saved results (the
full models' rows come from the final runs' JSONs, i.e. the numbers of experiments/RESULTS.md).

Usage (GPU, sequential, one process at a time):
  source env.sh && nice -n 10 $CPY experiments/transfer/train_variants.py            # trains what is missing, then renders
  $CPY experiments/transfer/train_variants.py --render                                # table only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtt import config as C  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "experiments" / "final_manifest.json"
RESULTS = Path(C.DATA_ROOT) / "experiments" / "results"
OUT_MD = REPO / "experiments" / "transfer" / "VARIANTS.md"
TAG_PREFIX = "transfer"
SPLIT_COL = "split_platform"
FINAL = {"size": ("gru", "S9"), "bev_ice": ("resnet", "B5")}          # final recipe (arch, recipe id) per task
LEAD = ["lead_valid", "lead_d_m", "lead_v_rel_mps", "lead_speed_mps", "lead_a_slope_mps2", "thw_s", "ttc_s"]
VARIANTS = {
    "V1": dict(name="lateral + speed", channels=["v_mps", "a_mps2", "standstill", "das_enabled", "steer_angle_deg", "steer_rate_degps",
                                                 "yaw_rate_radps", "curv_1pm", "lat_acc_mps2", "steer_pressed"],
               masked="lead group, steering torque, gas / brake pedal flags", datasets="commaSteeringControl"),
    "V2": dict(name="longitudinal-only", channels=["v_mps", "a_mps2", "standstill", "das_enabled", *LEAD],
               masked="every lateral channel (steering angle / rate / torque, yaw rate, curvature, lateral acceleration), steer_pressed, pedal flags",
               datasets="ULTra-AV, OpenACC platoons, UGA EV platoon, microSIM-ACC"),
    "V3": dict(name="speed-only", channels=["v_mps", "a_mps2", "standstill"],
               masked="everything but speed, acceleration and the standstill flag (no automation flag, no lead, no lateral channels, no pedals)",
               datasets="Yuan et al. Tesla Model 3, VED, OpenACC one-vehicle campaign"),
}
# diagnostics (size task, seed 0 only): attribute the V2 < V3 anomaly; `kwargs` override the recipe's kwargs
DIAGNOSTICS = {
    "V2nodro": dict(name="V2 channels, no group-DRO", channels=VARIANTS["V2"]["channels"], kwargs=dict(dro=False), tasks=("size",)),
    "V3das": dict(name="V3 + das_enabled", channels=[*VARIANTS["V3"]["channels"], "das_enabled"], kwargs={}, tasks=("size",)),
    "V3lead": dict(name="V3 + lead group", channels=[*VARIANTS["V3"]["channels"], *LEAD], kwargs={}, tasks=("size",)),
}
TASK_NAMES = {"size": "size (SMALL / MID / LARGE), bi-GRU recipe S9, 11 epochs", "bev_ice": "BEV vs ICE, 1-D ResNet recipe B5, 7 epochs"}
# one sentence per variant on what is lost, written after the runs (kept here so --render reproduces the page)
NOTES = {
    "V1": "Nothing is lost for the size task (54.5 / 66.1 / 59.5 window / drive / vehicle against 53.2 / 67.1 / 60.0; the "
          "human-only window score even rises by 6 points), so the size signal lives in the lateral channels (steering geometry "
          "at a given speed and yaw) and the lead, torque and pedal groups are dispensable for it. For BEV vs ICE the variant loses "
          "the pedal flags, i.e. the direct view of deceleration without a brake application (regenerative braking), and drops "
          "to 67.3 / 74.2 / 74.2 (-12.6 / -18.7 / -18.6); both classes suffer (drive-level recall BEV 71.6, ICE 77.5), so "
          "commaSteeringControl powertrain numbers must be read against roughly 74, not 93.",
    "V2": "For the size task the variant is at chance (34.8 / 33.4 / 33.2; drive-level recall MID 10.8, LARGE 22.1): without the "
          "lateral channels the recipe cannot learn size, and the lead group and the automation flag make it worse than the "
          "speed-only inputs (diagnostics below), so size results on longitudinal-only external sets are not interpretable with "
          "this variant; use V3 as the reference there. For BEV vs ICE most of the accuracy survives (73.0 / 88.8 / 89.8, -6.9 / "
          "-4.1 / -3.0): what is lost is BEV recall at the drive level (78.3 against 87.9, ICE unchanged at 97.1), the share of "
          "BEV drives whose regenerative deceleration is not visible without the pedal flags.",
    "V3": "Speed and acceleration alone keep a weak but real size signal (40.7 / 45.3 / 45.7, chance 33; SMALL 72.7, MID 37.2, "
          "LARGE 25.5 drive-level recall), and for BEV vs ICE they keep nearly everything the longitudinal set keeps (73.8 / 88.1 / "
          "91.0, -6.1 / -4.8 / -1.8; BEV recall 78.4, ICE 95.9): the lead group adds nothing to the powertrain decision, the loss "
          "against the full model is the missing pedal flags. Speed-only external sets (Yuan Tesla, VED, one-vehicle OpenACC) are "
          "therefore meaningful for the powertrain task and only indicative for size.",
}
READING = [
    "* The variants reuse the final recipes unchanged (same kwargs, same fixed epoch counts chosen for the 20-channel models, two "
    "seeds, no re-tuning), so they measure what the recipe loses with fewer inputs, not the best reduced-input model; the "
    "longitudinal size variants in particular were still improving at the last epoch.",
    "* External size results: compare V1 datasets with the full-model numbers (V1 is as good as the full model); for longitudinal "
    "or speed-only datasets use V3 and its 45-46 drive / vehicle reference, and do not use V2 for size.",
    "* External powertrain results: V2 and V3 datasets are compared with about 88-91 at drive and vehicle level (73-74 per window); "
    "V1 datasets with 74 (67 per window). Window-level numbers fall more than drive-level numbers in every variant, so "
    "external sets that only provide isolated 60-s windows (commaCarSegments, commaSteeringControl) are penalised twice.",
    "* The diagnostics show that under the model-disjoint split a channel can hurt: `das_enabled` and the lead group identify the "
    "driving-automation system and the lead sensor (radar or camera) of the training car models and let the network fit those "
    "models instead of the vehicle class, and group-DRO, which up-weights the worst-fitted car models, amplifies this. The "
    "same mechanism explains why the lateral channels hurt the powertrain variant V1 (steering geometry identifies the car model).",
]


def manifest_entry(task: str, arch: str, recipe_id: str, manifest: Path = MANIFEST) -> dict:
    man = json.loads(manifest.read_text())
    for e in man["entries"]:
        if e["kind"] == "window" and e["task"] == task and e["model"] == arch and e.get("recipe_id") == recipe_id and e["split_col"] == SPLIT_COL:
            return e
    raise KeyError(f"no window entry {task} / {arch} / {recipe_id} / {SPLIT_COL} in {manifest}")


def tag_of(task: str, vk: str, seed: int) -> str:
    arch, _ = FINAL[task]
    return f"{TAG_PREFIX}__{task}__{arch}__{vk}__s{seed}"


def final_tag(task: str, seed: int) -> str:
    arch, rid = FINAL[task]
    return f"{SPLIT_COL}__{task}__{arch}__{rid}__s{seed}"


def spec(vk: str) -> dict:
    return VARIANTS[vk] if vk in VARIANTS else DIAGNOSTICS[vk]


def plan(tasks: list[str], variants: list[str], seeds: list[int]) -> list[tuple[str, str, int, str, bool]]:
    return [(t, v, s, tag_of(t, v, s), (RESULTS / f"{tag_of(t, v, s)}.json").exists()) for t in tasks for v in variants for s in seeds
            if t in spec(v).get("tasks", tuple(FINAL))]


# ----------------------------------------------------------------------------- table
def _load(tag: str) -> dict | None:
    p = RESULTS / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _nums(r: dict) -> dict[str, float]:
    return dict(window=r["protocols"]["test_full"]["macro_f1"], balanced=r["protocols"]["test_balanced"]["macro_f1"],
                human=r["protocols"]["test_human_only"]["macro_f1"], drive=r["drive_level"]["test_full"]["macro_f1"],
                drive_human=r["drive_level"]["test_human_only"]["macro_f1"], vehicle=r["unit_level"]["vehicle_full"]["macro_f1"],
                val=r["val_macro_f1"])


def _cell(vals: list[float], ref: float | None = None) -> str:
    if not vals:
        return "-"
    m = 100 * float(np.mean(vals))
    s = f"{m:.1f}" if len(vals) == 1 else f"{m:.1f} ± {100 * float(np.std(vals)):.1f}"
    if ref is not None:
        d = m - 100 * ref
        s += f" ({'+' if d >= 0 else '−'}{abs(d):.1f})"
    return s


def render(tasks: list[str], variants: list[str], seeds: list[int], final_seeds: tuple[int, ...] = (0, 1, 2)) -> str:
    L = ["# Channel-subset variants of the final models (our data, held-out car models)", "", "Written 9 Nov 2026 by experiments/transfer/train_variants.py --render.", "",
         "Macro-F1 in percent on the `split_platform` test split (car models held out), mean ± std over seeds; in brackets the "
         "difference to the full model. `window` = one prediction per 60-s window (`test_full`), `drive` = mean class probability "
         "over the windows of a drive, `vehicle` = the same over all test drives of a vehicle (docs/PROTOCOLS.md). The full "
         "models are the final runs of experiments/final_manifest.json (3 seeds, the numbers of experiments/RESULTS.md); every "
         "variant re-trains the same recipe with the same fixed epoch count and kwargs, seeds "
         + ", ".join(map(str, seeds)) + ", with every channel outside its list set to NaN (= the missing value 0) on train, "
         "validation and test (`experiments.train.run(channels=...)`). Chance level is about 33 (size) and 50 (BEV vs ICE).", "",
         "| variant | inputs kept | masked |", "|---|---|---|"]
    for vk in variants:
        v = VARIANTS[vk]
        L.append(f"| {vk} {v['name']} ({len(v['channels'])} of 20) | `{', '.join(v['channels'])}` | {v['masked']}; external sets: {v['datasets']} |")
    L.append("")
    for task in tasks:
        L += [f"## {TASK_NAMES[task]}", "",
              "| model | inputs | window full | window balanced | window human-only | drive full | drive human-only | vehicle full | val | seeds |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        full = [r for r in (_load(final_tag(task, s)) for s in final_seeds) if r]
        ref = {k: float(np.mean([_nums(r)[k] for r in full])) for k in _nums(full[0])} if full else {}
        arch, rid = FINAL[task]
        if full:
            n = _nums(full[0])
            L.append(f"| full model ({arch} {rid}) | 20 | " + " | ".join(_cell([_nums(r)[k] for r in full]) for k in n) + f" | {len(full)} |")
        for vk in variants:
            rs = [r for r in (_load(tag_of(task, vk, s)) for s in seeds) if r]
            if not rs:
                L.append(f"| {vk} {VARIANTS[vk]['name']} | {len(VARIANTS[vk]['channels'])} | " + " | ".join(["-"] * 7) + " | 0 |")
                continue
            L.append(f"| {vk} {VARIANTS[vk]['name']} | {len(VARIANTS[vk]['channels'])} | "
                     + " | ".join(_cell([_nums(r)[k] for r in rs], ref.get(k)) for k in _nums(rs[0])) + f" | {len(rs)} |")
        L.append("")
        # per-class recall of the drive-level decision (what is lost per class)
        L.append("Per-class recall of the drive-level decision (`drive full`, mean over seeds):")
        L.append("")
        classes = list(full[0]["protocols"]["test_full"]["classes"]) if full else []
        if classes:
            L.append("| model | " + " | ".join(classes) + " |")
            L.append("|---|" + "---|" * len(classes))
            rows = [("full model", full)] + [(f"{vk} {VARIANTS[vk]['name']}", [r for r in (_load(tag_of(task, vk, s)) for s in seeds) if r]) for vk in variants]
            for name, rs in rows:
                if not rs:
                    continue
                rec = {c: [r["drive_level"]["test_full"]["per_class_recall"][c] for r in rs if r["drive_level"]["test_full"]["per_class_recall"].get(c) is not None] for c in classes}
                L.append(f"| {name} | " + " | ".join(f"{100 * np.mean(rec[c]):.1f}" if rec[c] else "-" for c in classes) + " |")
        L.append("")
    diag = [(dk, s) for dk in DIAGNOSTICS for s in (0,) if (RESULTS / f"{tag_of('size', dk, s)}.json").exists()]
    if diag:
        L += ["## Diagnostics (size task, seed 0): why longitudinal-only (V2) scores below speed-only (V3)", "",
              "Same recipe (S9) unless stated; one seed, so differences below about 2 points are noise. The last column is the "
              "training cross-entropy as the recipe weights it (group-DRO re-weights the samples, so the no-DRO run's value is "
              "not on the same scale).", "",
              "| run | inputs | window full | drive full | vehicle full | val | train cls loss (last epoch) |", "|---|---|---|---|---|---|---|"]
        rows = [(f"V2 {VARIANTS['V2']['name']}", [tag_of("size", "V2", s) for s in seeds]), (f"V3 {VARIANTS['V3']['name']}", [tag_of("size", "V3", s) for s in seeds])]
        rows += [(f"{dk} {DIAGNOSTICS[dk]['name']}" + (f" ({', '.join(f'{k}={v}' for k, v in DIAGNOSTICS[dk]['kwargs'].items())})" if DIAGNOSTICS[dk]["kwargs"] else ""),
                  [tag_of("size", dk, s)]) for dk, s in diag]
        for name, tags in rows:
            rs = [r for r in (_load(t) for t in tags) if r]
            if not rs:
                continue
            n = [_nums(r) for r in rs]
            L.append(f"| {name} | {len(spec(name.split()[0])['channels'])} | {_cell([x['window'] for x in n])} | {_cell([x['drive'] for x in n])} | "
                     f"{_cell([x['vehicle'] for x in n])} | {_cell([x['val'] for x in n])} | {np.mean([r['history'][-1]['parts']['cls'] for r in rs]):.3f} |")
        L.append("")
    L += ["## What each variant loses", ""]
    for vk in variants:
        if NOTES.get(vk):
            L.append(f"* **{vk} {VARIANTS[vk]['name']}**: {NOTES[vk]}")
    L += ["", "## Reading external results with these numbers", "", *READING]
    L += ["", "## Files", "",
          "* checkpoints: `<data-root>/experiments/ckpt/transfer__<task>__<arch>__<Vk>__s<seed>.pt`; `cfg[\"channels\"]` holds the "
          "variant's channel list (tensor order unchanged in `channels`), `experiments.regen_probs.load_checkpoint` rebuilds the "
          "network with its heads and a `Preprocess` that applies the same mask;",
          "* results: `<data-root>/experiments/results/<tag>.json` (`channels` and `cfg.channels` = the list) and `<tag>_probs.npz` "
          "(window_id, probs, y of `test_full`); `experiments.run_all.load_results` leaves these channel-restricted runs out of "
          "RESULTS.md and the paper tables;",
          "* script: `experiments/transfer/train_variants.py` (training and this page), training code: `experiments/train.py` "
          "(`channels`, `tag_prefix`), mask: `experiments/data.py` (`channel_keep`, `Preprocess(keep=)`, `Augment(keep=)`).", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="size,bev_ice")
    ap.add_argument("--variants", default="V1,V2,V3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and whether they are done")
    ap.add_argument("--render", action="store_true", help="only rewrite VARIANTS.md from the saved results")
    ap.add_argument("--no-render", action="store_true", help="train only")
    ap.add_argument("--diagnostics", action="store_true", help="also run the DIAGNOSTICS entries (size task, seed 0)")
    a = ap.parse_args(argv)
    tasks = [t for t in a.tasks.split(",") if t]
    variants = [v for v in a.variants.split(",") if v]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    bad = sorted(set(variants) - set(VARIANTS) - set(DIAGNOSTICS)) + sorted(set(tasks) - set(FINAL))
    if bad:
        raise SystemExit(f"unknown variants / tasks {bad}")
    todo = plan(tasks, variants, seeds)
    if a.diagnostics:
        todo += plan(["size"], list(DIAGNOSTICS), [0])
    render_variants = [v for v in variants if v in VARIANTS]
    if not a.render:
        for task, vk, seed, tag, done in todo:
            print(f"{tag}  {'done' if done else 'to run'}  epochs={manifest_entry(task, *FINAL[task])['fixed_epochs']} channels={len(spec(vk)['channels'])}")
        if a.dry_run:
            return 0
        if any(not done for *_, done in todo):
            import torch
            from experiments.data import WindowData
            from experiments.train import run
            data = WindowData()
            print(f"windows={len(data.win)} channels={len(data.channels)} device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
            for task, vk, seed, tag, done in todo:
                if done:
                    continue
                arch, rid = FINAL[task]
                e = manifest_entry(task, arch, rid)
                kwargs = dict(e.get("kwargs") or {})
                kwargs.update(spec(vk).get("kwargs") or {})
                print(f"start {tag} {time.strftime('%H:%M:%S')}", flush=True)
                run(task, SPLIT_COL, arch, seed, data=data, fixed_epochs=e["fixed_epochs"], eval_test=True, recipe_id=vk,
                    channels=list(spec(vk)["channels"]), tag_prefix=TAG_PREFIX, log=lambda m: print(m, flush=True), **kwargs)
    if not a.no_render:
        OUT_MD.write_text(render(tasks, render_variants, seeds))
        print(f"wrote {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
