# Window contract for the external transfer test (adapter authors read this first)

Code: `experiments/transfer/common.py` (contract, helpers) and `experiments/transfer/score.py` (scorer). Run everything
with `source env.sh && $CPY ...` from any directory; `common.py` puts the repo root on `sys.path`.

## What an adapter delivers

`save_windows(dataset, X, meta)` -> `<data-root>/external/<dataset>/work/windows.npz` + `meta.parquet`
(+ `windows_summary.json`). Nothing else is read by the scorer.

* `X`: float32 `(n, 600, 20)`, the 20 training channels in the order of `common.CHANNELS`
  (`v_mps, a_mps2, steer_angle_deg, steer_rate_degps, steer_torque_norm, yaw_rate_radps, curv_1pm, lat_acc_mps2,
  gas_pressed, brake_pressed, steer_pressed, standstill, das_enabled, lead_valid, lead_d_m, lead_v_rel_mps,
  lead_speed_mps, lead_a_slope_mps2, thw_s, ttc_s`), RAW PHYSICAL UNITS, NaN where the dataset has no value.
  Never clip, standardise or fill: `score.py` applies the training `Preprocess` (clip -> standardise with the training
  mean / std -> NaN -> 0).
* `meta`: one row per window with `dataset, vehicle_id, model, platform, drive_id, w_idx, t0_s, das_frac, moving_frac,
  footprint_m2, size_class, powertrain, borderline, channels_present, note`. `validate_windows` fills `size_class` and
  `borderline` from `footprint_m2`, `das_frac` / `moving_frac` / `channels_present` from `X`, normalises the powertrain
  vocabulary (`BEV / ICE / HEV / PHEV / FCEV / unknown`), and refuses windows below the moving rule.
* One `vehicle_id` = one physical car (the bootstrap clusters on it); one `drive_id` = one contiguous recording;
  `model` = the public car model name (public labels are fine; never put 16-hex device ids or route ids anywhere).

## Value conventions that must be reproduced (measured on the training tensors)

| channel(s) | convention |
|---|---|
| flags `gas_pressed brake_pressed steer_pressed standstill das_enabled lead_valid` | 0 / 1 floats, never NaN (NaN = "unknown" -> the standardised mean, which no training window ever had) |
| `lead_d_m lead_v_rel_mps lead_speed_mps lead_a_slope_mps2 thw_s ttc_s` | NaN whenever `lead_valid = 0` (60 % of training samples): that is "no lead", not a missing channel; `lead_v_rel` = lead minus ego; `lead_speed = v + v_rel`; `thw = d / v` (v > 0.5 m/s, d > 0.5 m); `ttc = d / closing` (closing > 0.1 m/s); slope = causal 1-s least-squares slope of the lead speed inside a lead episode, >= 3 samples, clipped +-4 -> `common.lead_channels()` does all of it |
| `yaw_rate_radps` | left-positive, from the device IMU / pose; `curv_1pm = yaw / v` only while v >= 1 m/s (NaN otherwise), `lat_acc_mps2 = v * yaw` -> `common.curvature_and_lat_acc()`; a steering-derived yaw (`common.yaw_from_steering`) is a documented deviation (say so in `note`) |
| `steer_angle_deg`, `steer_rate_degps` | steering wheel, left-positive, degrees; the rate is the CAN rate sample (a difference of the 10 Hz angle is a derived substitute) |
| `steer_torque_norm` | driver torque / 95th percentile of \|torque\| over moving samples with `steer_pressed`, per recording, NaN with < 50 such samples -> `common.torque_norm()` |
| `a_mps2`, `v_mps` | the vehicle's own aEgo / filtered CAN speed; `common.smooth_derivative(v)` is the derived fallback (noisier) |
| `das_enabled` | automation controller enabled OR stock cruise enabled (a lateral-only "engaged" flag or an ACC protocol flag is an approximation) |
| sampling | the 10 Hz rows are SAMPLES of the 100 Hz CAN stream (every 10th message), not bin averages: bring high-rate sources down with `resample_to_10hz(..., "decimate")` (or `"hold"`), use `"any"` for sparse pedal taps, `"linear"` for GNSS speed / positions, never `"mean"` for CAN signals |

Missing channels: a whole channel the dataset lacks is NaN everywhere. The lead, torque and pedal groups may be absent
without leaving the training distribution (the augmentation dropped them, with `lead_valid := 0`). Any other absent
channel (steering, yaw, standstill, das_enabled) is out of distribution: score with the matching reduced-input variant
(`score.py --variant ...`) or label the number "reduced input"; `score.py --mask-channels V2` applies the same mask to
the full model for comparison (`common.MASK_PRESETS`: V1 = lead + torque + pedals, V2 = longitudinal only, V3 = speed only).

## Window rule

600 consecutive 10 Hz samples, no gap (> 0.3 s between consecutive samples) after the first sample, >= 20 % of samples
with v >= 1 m/s. `cut_windows(df10hz, channel_map, t_col="t_s", info={...})` applies it to one recording:
`channel_map = {"v_mps": "Speed", "lead_d_m": gap_array, "steer_angle_deg": None, ...}` (column name, array, or None),
`align="run"` (default) restarts the blocks after every gap, `align="drive"` reproduces the lab's fixed 600-row blocks.
`concat_results([...])` stacks the drives. The trailing partial block is dropped; the stats dict says how many
candidates were lost to gaps / standstill.

## Scoring

```
$CPY experiments/transfer/score.py --dataset <name> [--task size|bev_ice|both] [--mask-channels lead,torque | V2]
                                   [--variant <name>] [--ckpts <glob>] [--bootstrap 1000] [--no-amp]
$CPY experiments/transfer/score.py --self-test --n 300
```
Default checkpoints: `split_platform__bev_ice__resnet__B5__s{0,1,2}` and `split_platform__size__gru__S9__s{0,1,2}`.
Outputs in `<dataset>/work/`: `probs__<checkpoint>__<mask>.npz` (aligned with `meta.parquet`), `probs__<set>__ens__<mask>.npz`
(mean over seeds), `score__<set>__<mask>.md` / `.json`. Levels: window, drive (mean probability per drive -> arg-max),
vehicle (all drives of a car pooled), model (all windows of a car model); accuracy with the 1 / K and majority chance
levels, per-class recall, macro-F1 when >= 2 classes are present, 95 % cluster-bootstrap intervals over vehicles.
Windows whose label is outside the task (HEV / PHEV / unknown) are scored and reported as predicted shares only.
Borderline models (|footprint - cut| < 0.05 m^2) are flagged and the size metrics are repeated without them.

## Pitfalls seen so far

1. Grid alignment: `0.01 * k != k / 10` in floating point; `resample_to_10hz` tolerates 1 us, your own code must too.
2. `lead_valid` NaN is not "no lead" (that is 0); `das_enabled` NaN is not "off" (that is 0).
3. The size task is about footprint, not length or mass: use the lab spec table's footprint where the model exists
   (`CANDIDATES.md` lists the values), public specs (+-0.05 m^2) otherwise, and flag borderline cars.
4. One car per model in most open sets: report per model (the scorer does) and cluster the bootstrap on vehicles, not
   windows; a single vehicle gives no interval.
5. ACC-driven recordings (ULTra-AV, OpenACC, microSIM) have `das_enabled = 1` by protocol: compare against the lab's
   automation-engaged windows, not the human-only protocol.
