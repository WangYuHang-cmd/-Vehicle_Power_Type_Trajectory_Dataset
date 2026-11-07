---
license: cc-by-4.0
task_categories:
  - tabular-classification
tags:
  - driving
  - vehicle-dynamics
  - CAN-bus
  - trajectories
  - time-series
  - driving-automation
  - sequence-classification
size_categories:
  - {{size_category}}
language:
  - en
pretty_name: "Vehicle_Power_Type_Trajectory_Dataset"
configs:
  - config_name: windows
    default: true
    data_files:
      - split: train
        path: "windows/train/*.parquet"
      - split: validation
        path: "windows/validation/*.parquet"
      - split: test
        path: "windows/test/*.parquet"
  - config_name: drives
    data_files:
      - split: train
        path: "index/drives.parquet"
  - config_name: windows_60s
    data_files:
      - split: train
        path: "index/windows_60s.parquet"
---

# Vehicle_Power_Type_Trajectory_Dataset (VTT)

**Can you tell what kind of vehicle produced a trace from how it moves?**

VTT is a sequence-classification dataset of trajectory and CAN-bus signals at {{hz}} Hz: ego speed and
acceleration, steering angle, steering rate and normalized steering torque, body yaw rate, pedal and turn-signal
flags, and the interaction with the lead vehicle (distance, relative speed, lead acceleration, headway, time to
collision). Each drive is labelled with the **footprint size class** and the **powertrain type** of the vehicle
that produced it. The release contains {{n_drives}} drives ({{total_hours}} h) from {{n_vehicles}} pseudonymous
vehicles on {{n_platforms}} vehicle platforms of {{n_brands}} brands, and {{n_windows_eligible}} eligible
{{window_s}} s windows. Periods during which a driving automation system (DAS) controlled the vehicle are kept and
labelled per sample ({{das_any_share_pct}} % of all samples show automation activity), so the data describe both
human driving and automated driving of the same vehicles.

No video, images, positions, timestamps or driver-monitoring signals are included.

## At a glance

| | |
|---|---|
| Drives (one ignition cycle each) | {{n_drives}} |
| Pseudonymous vehicles (device x platform) | {{n_vehicles}} |
| Identity groups (unit of all splits) | {{n_groups}} |
| Vehicle platforms / brands / models | {{n_platforms}} / {{n_brands}} / {{n_models}} |
| Hours (nominal) / hours moving | {{total_hours}} / {{moving_hours}} |
| Share of samples with DAS engaged / with any automation activity | {{das_share_pct}} % / {{das_any_share_pct}} % |
| {{window_s}} s windows: listed / eligible / human-only (strict) | {{n_windows_total}} / {{n_windows_eligible}} / {{n_windows_human_only_strict}} |
| Eligible windows per task: size / BEV-vs-ICE / four-class powertrain | {{n_windows_size_task}} / {{n_windows_bev_ice_task}} / {{n_windows_pt4_task}} |
| Windows in the ready-to-train shards | {{n_shard_windows}} in {{n_shards}} files |
| Size on disk: drive tables / window shards | {{data_gb}} GB / {{shards_gb}} GB |

## Labels

### Footprint size class (task A)

`size_class` is derived from the vehicle platform's **footprint = wheelbase x mean track width** (m^2) with two
fixed thresholds: `SMALL` below {{footprint_t1}} m^2, `MID` up to {{footprint_t2}} m^2, `LARGE` above. The
underlying specifications (`wheelbase_mm`, `track_front_mm`, `track_rear_mm`, `footprint_m2`, `curb_mass_kg`,
`body`, the legacy segment code `size_code`) are published per drive so that other size definitions can be
derived. `size_class` is null when the platform's specifications are unknown; such drives are not eligible for
task A (`task_size_eligible = False`).

### Powertrain (task B)

`powertrain` is one of `ICE`, `HEV`, `PHEV`, `BEV`, or null. `powertrain_raw` keeps the seven-value platform
code (`ICE`, `HEV`, `PHEV`, `BEV`, `MIXED`, `UNK`, `FCEV`); platforms sold with several powertrains that cannot be
told apart from the logs are `MIXED` and get a null `powertrain`. The **main task B is BEV vs ICE**
(`task_bev_ice_eligible = powertrain in {BEV, ICE}`); the four-class problem (`powertrain` not null) is a
secondary task. The two tasks are evaluated separately.

### Class support (eligible windows)

{{class_table}}

### Splits

Every drive carries two independent split assignments. The unit of both is the identity group (`group_id`: same
device, same vehicle identity or shared drive names), so no group appears in two splits.

* **`split_platform` (headline)**: whole car models (the lab taxonomy's base models, column `model`) are held out as the test split; a held-out model never
  appears in training or validation (about {{holdout_pct}} % of the hours of each size x powertrain cell,
  {{n_heldout_platforms}} held-out models, chosen so that a brand contributes both powertrain classes where
  possible). The remaining groups are split group-disjoint into `train` and `validation` (about {{val_pct}} % of
  the remaining hours), so a model may be shared between training and validation. Drives of a group that also
  drove a held-out model are marked `excluded` on its other models: they belong to no split and are left out
  of every protocol.
* **`split_device` (secondary)**: groups are assigned at random within each size x powertrain cell with
  {{train_pct}} / {{val_pct}} / {{test_pct}} % of the eligible hours; models may be shared across splits.

Lab-owned devices are always in `train`. A size x powertrain cell with fewer than {{low_support_hours}} test
hours, {{low_support_groups}} groups or {{low_support_platforms}} platforms is flagged `low_support_cell` and is
reported separately, never inside headline numbers.

{{split_table}}

### Protocols

All filters are boolean expressions over `index/windows_60s.parquet` columns (pandas syntax). Replace
`task_size_eligible` / `size_class` by `task_bev_ice_eligible` / `powertrain` for task B.

| protocol | filter |
|---|---|
| `train` | `eligible & task_size_eligible & (split_platform == "train")` |
| `validation` | `eligible & task_size_eligible & (split_platform == "validation")` |
| `test_full` (headline) | `eligible & task_size_eligible & (split_platform == "test")` |
| `test_balanced` | `window_id.isin(splits/eval_balanced_size_test.parquet.window_id)` (class-balanced, at most {{bal_group_pct}} % of a class from one group and {{bal_platform_pct}} % from one car model) |
| `test_human_only` | `test_full & human_only_strict` (no automation activity in the window nor in the {{human_guard_s}} s before it) |
| `split_device` variants | the same filters with `split_device` in place of `split_platform`; the balanced lists of this scheme are `splits/eval_balanced_<task>_test_device.parquet` |

Headline metric: macro-averaged F1 over the classes of the task on `test_full`, excluding windows with
`low_support_cell = True` (reported separately). `splits/train_uniform_<task>.parquet` provides a per-window
weight (inverse class frequency to the power {{cw_power}}, capped at {{cw_cap}}) for class-balanced training. The
balanced evaluation lists and both splits were generated with seed {{split_seed}}.

### Official model inputs

Under the official protocol a model may use only the following drive-table columns (the same columns form the
{{n_samples}}-step sequences of the window shards). Everything else in the drive tables and all index columns are
metadata for analysis and must not be fed to a model (they carry brand- and platform-specific shortcuts).

{{official_inputs}}

Not model inputs: {{nuisance_columns}}, and every column of the index tables.

### Reference results

Classifiers that use the official channels only (hand-crafted window statistics with logistic regression and gradient
boosting; a one-dimensional ResNet, a bidirectional GRU, a Transformer encoder and Mamba state-space models) are reported with
their code in `experiments/RESULTS.md` of the code repository, under exactly these protocols and split schemes. They
are reference points for the two tasks, not a leaderboard.

## Files

```
README.md                      this card
LICENSE                        CC BY 4.0
CHECKSUMS.sha256               sha256 of every other file (sha256sum -c compatible)
index/drives.parquet           one row per drive: labels, specifications, split assignments, provenance flags
index/vehicles.parquet         one row per pseudonymous vehicle
index/platforms.parquet        one row per vehicle platform with its specifications
index/windows_60s.parquet      one row per window (eligibility, statistics, labels, splits)
index/split_summary.csv        hours / windows / groups / platforms per task, class and split
index/readme_numbers.json      every number printed on this card
splits/splits.json             split configuration, per-cell support, group assignments, held-out platforms
splits/eval_balanced_<task>_<split>.parquet   window ids of the balanced evaluation lists (split_platform scheme)
splits/eval_balanced_<task>_<split>_device.parquet   the same lists for the split_device scheme
splits/train_uniform_<task>.parquet           per-window training weights (split_platform scheme)
splits/train_uniform_<task>_device.parquet    the same weights for the split_device scheme
schema/*.json                  machine-readable column descriptions (name, type, unit, description, source)
data/vehicle_NNNN/drive_NNNNNN.parquet        full drive tables, {{hz}} Hz, one row per sample
windows/{train,validation,test}/*.parquet     ready-to-train window shards (index columns + sequences)
```

### Drive tables (`data/`)

Row `k` is the `k`-th sample of the vehicle-state stream; `t_s = k / {{hz}}` is nominal time. A `gap` flag marks
rows preceded by more than {{gap_s}} s without data. Lower-rate streams (lead vehicle, yaw rate, automation
state) are held causally to the sample times and set to NaN when older than {{lead_max_age_s}} s (lead) or
{{pose_max_age_s}} s (yaw).

{{field_table}}

### Window shards (`windows/`)

One row per eligible window that is eligible for at least one task, assigned to `train` / `validation` / `test`
by `{{split_col}}`. Columns: every column of `index/windows_60s.parquet` plus one `{{n_samples}}`-element float32
sequence per official input (booleans as 0 / 1). Files are at most {{shard_max_mb}} MB each.

{{shard_table}}

### Index tables

`index/drives.parquet`

{{drives_index_table}}

`index/windows_60s.parquet`

A window is `eligible` when it has {{n_samples}} consecutive samples, no gap, and the vehicle is moving
(v >= {{moving_v}} m/s) for at least {{min_moving_pct}} % of the samples.

{{windows_index_table}}

`index/vehicles.parquet`

{{vehicles_table}}

`index/platforms.parquet`

{{platforms_table}}

## Limitations and caveats

* **Labels are platform attributes.** `size_class` and `powertrain` describe the vehicle platform identified from
  the vehicle's own bus identification, not a measurement of each drive. A misidentified platform mislabels every
  drive of that vehicle; known label-noise notes are in `label_note`.
* **MIXED platforms are null.** Platforms sold with several powertrains that the logs cannot separate carry no
  `powertrain` label ({{pt_null_pct}} % of the drives) and are excluded from task B.
* **Mild hybrids** follow the platform code of their generation and may be labelled `ICE`.
* **Automation share.** {{das_any_share_pct}} % of the samples show DAS activity; control style during these periods
  reflects the controller as much as the vehicle. Use `das_enabled`, `das_any`, `human_only` and `human_only_strict`
  to restrict analyses.
* **Lead-vehicle source.** On {{radar_unavailable_pct}} % of the drives the lead vehicle comes from a camera model
  rather than a radar (`radar_unavailable`, `lead_source`); distance noise and availability differ between the two.
* **Torque units.** `steer_torque_raw` is in each make's own CAN units and is not comparable across brands;
  `steer_torque_norm` divides by the drive's {{torque_q_pct}}th percentile of |torque| while moving and steering.
* **Dead channels.** The pedal magnitudes are not populated on every vehicle: `gas` is NaN on {{gas_dead_pct}} % and
  `brake` on {{brake_dead_pct}} % of the drives (`gas_alive`, `brake_alive`); the `*_pressed` flags are always valid.
* **Quantization and sampling.** Signal resolution and the exact sample clock differ by make; tables are indexed by
  sample, not by a fixed clock, and `t_s` is nominal.
* **Region and software confounds.** `region`, `software_era`, `device_type`, `pose_source` and the lead source are
  correlated with brand and powertrain; they are metadata, not inputs.
* **Low-support cells.** Some size x powertrain cells have few groups or platforms; they are flagged
  `low_support_cell` and excluded from headline metrics.
* **Duplicates.** Drives stored twice were merged before pseudonymisation; drives of one device remain correlated,
  which is why groups, not drives, are the unit of the splits.
* **Car-following definitions.** `thw_s` requires v > {{thw_v_min}} m/s; the car-following share `cf_frac` uses
  headway <= {{thw_cf_s}} s.

## Privacy and ethics

* Vehicles, drives and groups carry random, independent pseudonyms drawn for this release only; they cannot be
  linked to identifiers of any other dataset.
* The release contains no positions, no absolute times or dates, no video or images, and no driver-monitoring
  signals. `region` is a continent-level bucket; `seg` is an ordinal; `t_s` is relative to the start of the drive.
* Labels describe vehicle models, not people. Rare models may be recognisable as a model, never as a person.
* Access is granted for research on vehicle dynamics and driving behaviour. **Re-identification of vehicles or
  drivers and linkage with other datasets are prohibited.**
* Questions, corrections or removal requests: {{contact_email}}.

## Citation

If you use VTT, please cite the paper below ([arXiv:2603.06986](https://arxiv.org/abs/2603.06986)):

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

## License

The data tables are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`).
The build and evaluation code is released under the MIT license at `{{github_repo}}`. Dataset repository:
`{{hf_repo}}`; release `{{release_version}}`, table schema `{{schema_version}}`, build configuration
`{{config_version}}`.
