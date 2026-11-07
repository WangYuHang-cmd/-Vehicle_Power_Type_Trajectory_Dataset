# Evaluation protocols, reference features and shortcut audits

This page is the contract between the dataset card ("Protocols" section), `starter_kit/eval.py`,
`starter_kit/features.py` and the two audits of `vtt/audits/`. Column names refer to `vtt/schema.py`.

## Tasks

| task | label column | classes | eligibility column |
|---|---|---|---|
| `size` | `size_class` | SMALL / MID / LARGE | `task_size_eligible` |
| `bev_ice` (main powertrain task) | `powertrain` | BEV / ICE | `task_bev_ice_eligible` |
| `powertrain4` (secondary) | `powertrain` | ICE / HEV / PHEV / BEV | label not null |

## Split columns

* `split_platform` (headline): no car model (taxonomy base model, `config.HOLDOUT_KEY`) is shared between train and the evaluation splits.
* `split_device`: identity groups are disjoint across splits; car models may be shared.

Both are columns of the drives and windows indexes; a group is in exactly one split per column. The balanced
evaluation lists exist once per scheme: `splits/eval_balanced_<task>_<split>.parquet` belongs to `split_platform`
(headline) and `splits/eval_balanced_<task>_<split>_device.parquet` to `split_device` (likewise
`splits/train_uniform_<task>[_device].parquet` for the training weights); `starter_kit.load.load_balanced_ids(task,
split, split_col=...)` / `balanced_list_path` pick the file of the chosen scheme.

## Protocols (`starter_kit.eval.protocol_mask`)

| protocol | filter on the windows index |
|---|---|
| `test_full` (headline) | `eligible & <task eligibility> & (<split column> == "test")` and a class label of the task |
| `test_balanced` | `test_full & window_id.isin(splits/eval_balanced_<task>_test[_device].parquet.window_id)` (class-balanced, per-group and per-platform caps; the list of a scheme is already a subset of that scheme's `test_full`, so the intersection only guards against mixing the two schemes: ids outside `test_full` are dropped with a warning) |
| `test_human_only` | `test_full & human_only_strict` (no automation activity in the window nor in the 10 s before it) |

The same masks with `split = "validation"` or `"train"` select the other splits (`test_full` then simply means
"every eligible window of the task").

## Metrics (`starter_kit.eval.evaluate`)

* `macro_f1_supported` (headline): macro-averaged F1 over the classes present in the selection, computed on the
  windows outside `low_support_cell` cells.
* `macro_f1`: the same over all selected windows; `accuracy`.
* Chance levels: `majority_f1` (always the majority class) and `permuted_f1` (mean over label permutations).
* `per_cell`: accuracy inside every `size_class x powertrain` cell with its low-support flag.
* Confusion matrix (rows = true, columns = predicted) written as CSV when an output directory is given.

Predictions are passed as a `pandas.Series` indexed by `window_id`, a dict, or an array aligned with the selection.

## Reference features (`starter_kit.features.window_features`)

All features are computed from one 600-sample window slice of a drive table; units m/s, m/s^2, deg, rad/s, 1/m, s.

| group | features | definition |
|---|---|---|
| speed | `v_mean, v_max, v_std, moving_frac, standstill_frac, a_std, a_p95, a_p05` | moving = `v_mps >= 1`; acceleration quantiles over moving samples |
| episodes | `n_accel_ep, accel_dv_mean, accel_mean_a, accel_peak_a, accel_t_peak_s, accel_plateau_frac` and the `decel_*` twins | episodes from `vtt.features_core` (config `ACCEL_EP`: SG-smoothed speed monotone within 0.3 m/s, >= 3 s, |dv| >= 2 m/s, mean |a| >= 0.1); peak = max |a| of the smoothed speed, time to peak from the episode start, plateau = share of the episode with |a| >= 75 % of the peak |
| lift-off | `coast_decel_mean, coast_frac` | mean `a_mps2` while `v > 3`, no pedal pressed, no automation (`das_any` False), not standstill and `a < -0.05` |
| steering | `torque_abs_mean, torque_abs_p95, steer_pressed_frac, steer_rate_abs_mean, steer_angle_abs_mean, k_steer_degm` | over moving samples without automation steering; `k_steer_degm` = least-squares slope through the origin of `steer_angle_deg` against `curv_1pm` (v > 3 m/s, valid yaw, |curvature| >= 1e-3) |
| yaw | `yaw_abs_mean, yaw_abs_p95, curv_abs_mean, lat_acc_abs_p95, yaw_valid_frac` | over moving samples with `yaw_valid` |
| lead | `lead_frac, cf_frac, thw_med, thw_p10, ttc_min, lead_speed_mean, lead_d_mean, lead_vrel_std` | `cf_frac` = share of valid-lead samples with `thw_s <= 6` |
| catch-up | `n_catchup_events, catchup_responded_frac, catchup_delay_mean_s, catchup_gap_change_5s_mean_m, catchup_accel_ratio_mean` | onsets where the lead slope is >= 0.5 m/s^2 for 1 s after 2 s of |slope| < 0.3 (config `LEAD_ACCEL_EVENT`), allowing up to 0.5 s of transition between the quiet window and the hold (the windows index counts onsets without that allowance), restricted to car-following range (`thw_s <= 6`) and, by default, to onsets without automation; delay = first time `a_mps2 >= 0.3` holds 0.5 s within 8 s; gap change = `lead_d_m` 5 s after the onset minus at the onset; accel ratio = mean ego acceleration over the 2 s after the response divided by the mean lead slope over the hold |

`features_table(windows, load_drive)` evaluates them for every row of a windows index (one read per drive).

## Shortcut audits (`vtt/audits`, `scripts/s12_audits.py`)

Deliberately simple numpy classifiers under group-disjoint cross-validation (groups = `group_id`), with the
macro-F1 compared against two chance levels (always-majority, label permutation):

* **Nuisance metadata -> labels / splits** (`vtt/audits/nuisance.py`): predictors `device_type`, `software_era`,
  `region`, `pose_source`, `radar_unavailable`, `op_long`, `gas_alive`, `brake_alive`, `torque_alive`, `has_pose`,
  `rlog_dec_any` (one drive = one row); Cramer's V per predictor plus a per-value majority vote for every single
  predictor and for the joint key `ALL`; targets `powertrain`, `size_class`, `split_platform`, `split_device`.
* **Official-input statistics -> brand** (`vtt/audits/inputs_brand.py`): the window-index summary statistics
  (`v_mean` ... `brake_pressed_frac`) predict `brand`, `platform`, `powertrain` and `size_class` with a nearest
  centroid on standardised statistics and a quantile-binned majority vote per statistic.

Outputs under `build/qc/`: `audit_nuisance.csv`, `audit_inputs_brand.csv`, `audit_numbers.json`, `AUDITS.md`
(the bullets for the dataset card's limitations section).
