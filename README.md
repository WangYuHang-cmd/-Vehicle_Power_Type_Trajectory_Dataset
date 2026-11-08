# Vehicle_Power_Type_Trajectory_Dataset (VTT): build pipeline and starter kit

Code that builds, checks and publishes **VTT**, a sequence-classification dataset of trajectory and CAN-bus
signals (10 Hz) for two tasks: vehicle **size class** (footprint = wheelbase x mean track) and **powertrain**
(BEV vs ICE as the main task; ICE / HEV / PHEV / BEV as a secondary task). Periods in which a driving automation
system (DAS) controlled the vehicle are kept and labelled per sample, so the data describe both human and automated
control. No video, no images, no location data, no timestamps and no driver-monitoring streams are released.

* Dataset (tables, CC BY 4.0, gated): `https://huggingface.co/datasets/HenryYHW/Vehicle_Power_Type_Trajectory_Dataset`
* Code (this repository, MIT): `https://github.com/WangYuHang-cmd/-Vehicle_Power_Type_Trajectory_Dataset`
* Related paper / citation token: ADAS-TO (arXiv:2603.06986), see the end of this file.

## Clone

The repository name starts with `-`, so always give `git` an explicit target directory:

```bash
git clone 'https://github.com/WangYuHang-cmd/-Vehicle_Power_Type_Trajectory_Dataset.git' vtt
cd vtt
```

## Layout

| path | content |
|---|---|
| `vtt/` | the build package: `config.py` (every switch), `paths.py`, `schema.py` (public tables), `private.py`, labels, pseudonyms, drives, windows, QC, splits, package, `gate/` (release and code gates), `audits/` |
| `scripts/` | one script per stage (`s01` ... `s12`), each with `--data-root`, `--dry-run` where relevant, exit code 2 on a failed check |
| `starter_kit/` | `load.py` (Hub or local folder), `features.py` (reference window features), `eval.py` (official protocols and metrics) |
| `experiments/` | the classification experiments on the released windows: `data.py` (tensor cache, protocol masks), `models.py` (1-D ResNet, bi-GRU, Transformer encoder, Mamba-2 state-space-duality blocks and Mamba-1 selective scan in pure PyTorch), `train.py`, `run_all.py`, `features_baseline.py`; results in `experiments/RESULTS.md` |
| `sample/` | a synthetic miniature release (two fictitious vehicles, two drives each, balanced lists of both split schemes) so that the starter kit and the tests run without the dataset |
| `tests/` | pytest suites (`*_tests.py`), synthetic data only |
| `docs/` | `PIPELINE.md` (stage table and file contracts), `PROTOCOLS.md` (evaluation protocols, features, audits), the dataset card template |

## Environment

Two interpreters are used: the build and the tests run on a Python 3.12 environment with `numpy >= 2`,
`pandas >= 2.2`, `pyarrow >= 17` and `pytest` (no scipy, no scikit-learn); the upload step and the starter kit
also work with `huggingface_hub` and, optionally, `datasets` (streaming check) and `torch`. Install the
requirements with `pip install -r requirements.txt`.

Every path is taken from environment variables (see `vtt/config.py`); set them before running anything. The
public export of this repository (`scripts/s11_github_export.py`) replaces the maintainers' local absolute paths in
every text file, `vtt/config.py` defaults included, by the shell-style placeholders below, so a default that still
reads `"$VTT_DATA_ROOT"` or `"$PY"` simply means "set this variable":

| variable / placeholder | meaning |
|---|---|
| `VTT_DATA_ROOT` (`$VTT_DATA_ROOT`, `$D`) | data root with `private/` (mode 0700, never published), `work/`, `build/`, `release/<version>/`, `tmp/`, `hf_home/` |
| `VTT_RAW_ROOT` (`$VTT_RAW_ROOT`) | the raw log library (read-only) |
| `VTT_TOCF_ROOT` (`$VTT_TOCF_ROOT`) | checkout of the upstream extractor package reused unchanged for stage S2 |
| `VTT_REUSE_EXISTING_SIGNALS`, `VTT_SEED_PRIVATE_ROOT` (`$VTT_TOCF_DATA`) | an earlier extraction of the same extractor whose signal directories S2 hardlinks, and the private seed (salt) that makes its uids match |
| `VTT_TAXONOMY_DIR`, `VTT_ROUTE_MASTER` (`$VTT_TAXONOMY_DIR`, `$VTT_ROUTE_MASTER`, `$VTT_OVERVIEW_DIR`) | label sources: vehicle taxonomy folder and route master table (both under the maintainers' overview folder) |
| `VTT_OPENDBC_ROOT`, `VTT_OPENDBC_LOCAL` (`$VTT_OPENDBC_LOCAL`) | upstream `opendbc` clone used for vehicle specifications, and an older local copy used as fallback |
| `VTT_VENV_PY`, `VTT_CONDA_PY` (`$PY`, `$CPY`) | the build / test interpreter and the interpreter with `huggingface_hub` for `s10_upload.py` and the starter kit |
| `$R`, `$VTT_PROJECTS_DIR` | this repository's root (`PYTHONPATH`) and the folder that contains it |
| `$VTT_DATA_PARENT`, `$VTT_DATASETS_DIR`, `$HOME` | the parents of the data root and of the raw library, and the maintainer's home directory (appear only in comments and examples) |
| `HF_HOME`, `HF_XET_CACHE`, `TMPDIR` | caches; keep them on a disk with room |

The extractor itself is not vendored: stage S2 runs the upstream `s02_extract_canonical.py` (version `s02-v4`)
with the VTT roots; its output contract is documented in `docs/PIPELINE.md`.

## Stage order

| stage | script | what it does |
|---|---|---|
| S1 | `s01_probe_uidmap.py` | route manifest, probe, uid map, route lists and chunks |
| S2 | `s02_extract_all.sh`, `s02c_segments_table.py`, `s02d_accept.py` | canonical extraction (upstream extractor), segments table, acceptance |
| S2b | `s02b_audit_probe.py` | label-audit probe (private) |
| S3 | `s03_platforms.py`, `s03_labels.py`, `s03_inventory.py` | platform mapping, specs, labels, size inventory |
| S4 | `s04_pseudonyms.py` | vehicle / drive / group pseudonyms (fresh namespace) |
| S5 | `s05_build_drives.py` | public drive tables (sample-indexed 10 Hz) and the drives index |
| S6 | `s06_windows.py` | 60 s windows (600 samples) and the windows index |
| S7 | `s07_qc.py` | quality control and label review |
| S8 | `s08_splits.py` | device-disjoint and platform-disjoint splits, balanced evaluation lists |
| S9 | `s09_package.py`, `s09b_gate.py` | release tree, README, checksums, release gate |
| S10 | `s10_upload.py` | Hub upload (`readme-validate`, `upload-data`, `readme`, `check`, `settings`) |
| S11 | `s11_github_export.py` | this repository's public export (path rewrite, sample, code gate, push commands) |
| S12 | `s12_audits.py` | shortcut audits (nuisance metadata -> labels / splits; official inputs -> brand) |

Run `python scripts/<script>.py --help` for the options. All scripts are idempotent and print counts only.

## Starter kit

```python
from starter_kit import load, features, eval as ev

root = "sample"                                   # or None to read the Hub repository (gated: `hf auth login` first)
windows = load.load_index("windows", root=root)
drives = load.load_index("drives", root=root)
for row, x in load.iter_windows("test", "test_full", "bev_ice", root=root, limit=2):
    print(row["window_id"], x.shape)              # (600, 20): the official inputs, booleans as 0 / 1
feats = features.features_table(windows, lambda d: load.load_drive(d, root=root, drives=drives))
result = ev.evaluate(windows, {w: "ICE" for w in windows.window_id}, task="bev_ice", protocol="test_full")
print(result["macro_f1_supported"], result["n_windows"])
```

Protocols (`test_full`, `test_balanced`, `test_human_only` on `split_platform` as headline and `split_device`; the
balanced lists are `splits/eval_balanced_<task>_<split>.parquet` and, for `split_device`, `..._device.parquet`),
metrics and the reference features are described in `docs/PROTOCOLS.md`.

## Tests

```bash
python -m pytest -q
```

The tests use synthetic data only; nothing in this repository or its tests contains a real device, route or
vehicle identifier, and the code gate (`vtt/gate/pre_push.py`) scans every commit for such tokens before a push.

## Privacy

Pseudonyms (`vehicle_NNNN`, `drive_NNNNNN`, `group_NNNN`) live in a fresh namespace in random order and cannot be
linked to other datasets. The mapping, the salt and every input with real identifiers stay under the private root,
which is never exported or uploaded. See the dataset card for the full privacy and ethics statement.

## License

Code: MIT (`LICENSE`). Data tables: CC BY 4.0 (dataset card). Third-party notices: `THIRD_PARTY_NOTICES.md`.

## Citation

```bibtex
@article{wang2026adasto,
  title   = {ADAS-TO: A Large-Scale Multimodal Naturalistic Dataset and
             Empirical Characterization of Human Takeovers during ADAS Engagement},
  author  = {Wang, Yuhang and Xu, Yiyao and Sun, Jingran and Zhou, Hao},
  journal = {arXiv preprint arXiv:2603.06986},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.06986}
}
```

## Experiments

`experiments/` trains sequence classifiers on the 60-s windows using the official input channels only. The tensor
cache is written once from the release tables (`scripts/s13_export_tensors.py`); labels, splits and protocols are
joined from the index tables, so every number in `experiments/RESULTS.md` is reproducible from the published dataset.

```bash
python -m experiments.features_baseline                      # hand-crafted statistics + logistic regression / gradient boosting
python -m experiments.run_all --models transformer,gru,resnet,mamba2
python -m experiments.train --task bev_ice --split-col split_platform --model mamba2   # one configuration
```

The method-research campaign that followed the first round is documented in `experiments/CAMPAIGN.md` (pre-registration,
gates and log), `experiments/ORACLES.md` (zero-training ceilings: aggregation rules, ensembles, vehicle pooling, evidence
sufficiency, footprint-noise and curb-mass oracles), `experiments/INTERNAL_FOLDS.md` (model-disjoint folds used for every
selection), `experiments/RECIPE.md` (training-recipe sweep on the folds), `experiments/AGGREGATION.md` (drive- and
vehicle-level aggregation study) and the drive- / vehicle-level protocols in `docs/PROTOCOLS.md`. The episode-centric
multiple-instance model lives in `experiments/episodes.py`, `episodes_build.py`, `window_embed.py`, `mil_data.py`,
`mil_models.py` and `mil_train.py`; `experiments/eval_final.py` is the frozen final evaluation over `final_manifest.json`.

Requirements beyond the build environment: PyTorch (CUDA recommended) and scikit-learn. `experiments/ANALYSIS.md`
(from `python -m experiments.analysis`) breaks the test predictions down by held-out car model, by size x powertrain
cell and by distance of the footprint to the class thresholds.
`experiments/LLM_FEATURES.md` reports the LLM-assisted feature discovery: agents proposed physics-based window
features with code (`experiments/llm_features/`, one module per feature, built by `llm_features_build.py` from the
verified proposals), which were computed for every window (`llm_features_compute.py`) and evaluated with the same
classifiers and protocols as the plain statistics (`llm_features_eval.py`).

## Paper

`paper/` holds the IEEE IV 2027 manuscript draft: `main.tex` (anonymous build by default; the author build is
`pdflatex -jobname=main_authors "\def\withauthors{1}\input{main}"`), the sections, `refs.bib`, the figures and the
generated tables, plus `OUTLINE.md`, the single source of every number quoted in the text (each traced to the
experiment reports), and `check_draft.py`, which scans the sources for identifiers and wording rules. Compiled PDFs:
`paper/main.pdf` (anonymous) and `paper/main_authors.pdf`.
