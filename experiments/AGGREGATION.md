# Aggregation study: from window probabilities to drive- and vehicle-level decisions

How the per-window class probabilities of a trained window model are turned into one decision per drive (`drive_full`,
`drive_human_only`) and per vehicle (`vehicle_full`, `vehicle_full_multi`), see docs/PROTOCOLS.md for the unit
protocols. Everything is computed from saved window probabilities by `experiments/aggregate.py`; no model is trained here.

Rules compared (per unit; ties go to the earlier class in task order):

- `mean`: weighted mean of the probability vectors (the published `drive_level` rule with uniform weights);
- `logodds`: weighted mean of log probabilities, mapped back through a softmax (geometric mean, insensitive to a
  temperature);
- `median`: per-class (weighted) median, renormalised; `vote`: weighted share of per-window argmax votes;
- `mean_T`: mean of temperature-scaled probabilities, the temperature fitted by negative log-likelihood out of sample;
- `gate_mean` / `gate_logodds`: mean / log-odds mean with weights = out-of-sample probability that the window is
  classified correctly (logistic regression on window statistics: automation share, episode and lead-event counts,
  pedal / steering shares, speed and acceleration statistics, confidence and entropy of the window);
- `fp_mean` (size task, runs that carry a footprint estimate): mean footprint over the windows, cut at the class
  thresholds.

Weights: `das{a}_ep{b}_steer{c}` = (1 - das_any_frac + 0.1)^a (1 + n_accel_ep + n_decel_ep)^b (steer_pressed_frac + 0.05)^c
with a, b, c in {0, 0.5, 1}; `uniform` is (0, 0, 0).

Vehicle levels (`vehicle`, `vehicle_multi`): every rule is applied in two ways. Applied to the pooled windows of the
vehicle (method name unchanged), `mean / uniform` is exactly the protocol `vehicle_full` / `vehicle_full_multi` of
docs/PROTOCOLS.md (all the vehicle's evaluated windows with the same weight) and is the reference the deltas are
measured against. Applied in two stages (method name with the suffix `>drives`), the rule's drive decisions are
combined by a log-odds mean weighted by n_windows^0.5 (the candidate rule of the plan: a vehicle with many short drives
is not dominated by its single longest one); it is a candidate, not the protocol. `vehicle_multi` keeps the vehicles
with at least two drives among the evaluated windows. The vehicle levels under protocol `human_only` are exploratory
(no published protocol pools human-only windows per vehicle). Stacking: multinomial logistic regression on the centred
log probabilities of several runs, fitted out of sample.

Which runs are compared: on the out-of-fold path only the pooled fold runs of a configuration (all folds of one task x
model x recipe x seed, kind `oof_pooled`) are tabulated and stacked; single-fold sweep runs, the leave-one-car-model-out
probes (recipe ids `L<k>_...`, whose windows are one car model and disjoint from the fold windows) and other kinds are
left out unless asked for (`--kinds`, `--recipes`, `--include-probes`). Stacking members must share their windows: a
member whose windows are covered by less than half by the common window set is dropped (`--min-cover`).

Out-of-sample means: by fold when the fold runs of one configuration are pooled, otherwise leave-one-car-model-out
inside the evaluated windows (the temperature, gate or stacker applied to a car model never saw that model). Numbers in
a section marked post-hoc come from the published models' test-set predictions: they show that the code works and the
size of the effects, but choosing a rule from them would be optimistic; the rule used for the final evaluation is
chosen on the internal folds only (experiments/CAMPAIGN.md).

<!-- rendered by experiments/aggregate.py: everything below this line is overwritten -->

## Size: SMALL / MID / LARGE (split_platform) - internal folds (out-of-fold)

Runs: 2 (gru, resnet); kinds: oof_pooled; recipes: S0; macro-F1 in percent, mean over runs; `delta` = mean paired difference to `mean / uniform` (the protocol rule: mean over the drive's windows at the drive level, mean over the vehicle's pooled windows at the vehicle levels), `improved` = runs with a positive difference; `mean>drives / uniform` is the plan's two-stage vehicle candidate (log-odds mean of the drive decisions, n_windows^0.5 weights).

### Reference rule vs best rule per level and protocol

| level | protocol | mean / uniform (protocol) | mean>drives / uniform | best rule | macro-F1 | delta | improved | units |
|---|---|---|---|---|---|---|---|---|
| drive | full | 67.5 | - | mean_T / uniform | 67.8 | +0.3 | 2/2 | 3050 |
| vehicle | full | 60.5 | 61.4 | mean>drives / das0_ep1_steer0 | 61.6 | +1.0 | 2/2 | 610 |
| vehicle_multi | full | 61.9 | 63.5 | gate_mean>drives / gate | 63.5 | +1.6 | 2/2 | 341 |
| drive | human_only | 61.8 | - | gate_mean / gate | 62.0 | +0.2 | 1/2 | 1935 |
| vehicle | human_only | 59.4 | 59.4 | logodds / das0_ep1_steer1 | 60.5 | +1.0 | 2/2 | 520 |
| vehicle_multi | human_only | 60.0 | 59.9 | logodds / das0_ep1_steer1 | 61.8 | +1.8 | 2/2 | 279 |

### Drive level, protocol full: ranked rules (top 10 plus every unweighted rule)

| method | weights | macro-F1 | std over runs | delta | improved | runs |
|---|---|---|---|---|---|---|
| mean_T | uniform | 67.8 | 0.8 | +0.3 | 2/2 | 2 |
| gate_logodds | gate | 67.8 | 1.0 | +0.2 | 1/2 | 2 |
| logodds | das0_ep0.5_steer0 | 67.7 | 0.4 | +0.1 | 1/2 | 2 |
| gate_mean | gate | 67.6 | 1.0 | +0.0 | 1/2 | 2 |
| logodds | uniform | 67.6 | 0.6 | +0.0 | 1/2 | 2 |
| mean | uniform | 67.5 | 0.7 | +0.0 | 0/2 | 2 |
| mean | das0.5_ep0_steer0 | 67.5 | 0.2 | -0.0 | 1/2 | 2 |
| mean | das0_ep0_steer0.5 | 67.5 | 0.3 | -0.1 | 1/2 | 2 |
| mean | das0_ep0.5_steer0 | 67.5 | 0.6 | -0.1 | 1/2 | 2 |
| logodds | das0.5_ep0_steer0 | 67.4 | 0.3 | -0.1 | 1/2 | 2 |
| vote | uniform | 67.1 | 0.4 | -0.5 | 0/2 | 2 |
| median | uniform | 66.7 | 0.5 | -0.9 | 0/2 | 2 |

### Vehicle level, protocol full: ranked rules (top 10 plus every unweighted rule)

| method | weights | macro-F1 | std over runs | delta | improved | runs |
|---|---|---|---|---|---|---|
| mean>drives | das0_ep1_steer0 | 61.6 | 0.3 | +1.0 | 2/2 | 2 |
| mean | das0_ep1_steer0 | 61.5 | 0.4 | +1.0 | 2/2 | 2 |
| mean_T>drives | uniform | 61.5 | 1.0 | +0.9 | 2/2 | 2 |
| mean | das0_ep0.5_steer0.5 | 61.5 | 1.4 | +0.9 | 1/2 | 2 |
| mean>drives | uniform | 61.4 | 0.4 | +0.9 | 2/2 | 2 |
| mean>drives | das0_ep1_steer0.5 | 61.4 | 1.7 | +0.8 | 1/2 | 2 |
| mean>drives | das0.5_ep0_steer0 | 61.4 | 0.5 | +0.8 | 2/2 | 2 |
| mean>drives | das0_ep0.5_steer0.5 | 61.4 | 1.2 | +0.8 | 1/2 | 2 |
| logodds>drives | das0_ep1_steer0 | 61.3 | 0.1 | +0.8 | 2/2 | 2 |
| mean>drives | das0_ep0_steer0.5 | 61.3 | 0.4 | +0.8 | 2/2 | 2 |
| gate_mean>drives | gate | 61.2 | 0.5 | +0.7 | 2/2 | 2 |
| mean_T | uniform | 61.0 | 0.7 | +0.5 | 1/2 | 2 |
| gate_logodds | gate | 60.9 | 0.2 | +0.4 | 2/2 | 2 |
| vote>drives | uniform | 60.8 | 1.4 | +0.3 | 1/2 | 2 |
| gate_mean | gate | 60.8 | 0.1 | +0.3 | 2/2 | 2 |
| gate_logodds>drives | gate | 60.8 | 0.7 | +0.3 | 1/2 | 2 |
| median>drives | uniform | 60.7 | 0.6 | +0.2 | 1/2 | 2 |
| mean | uniform | 60.5 | 0.1 | +0.0 | 0/2 | 2 |
| logodds | uniform | 60.5 | 0.6 | -0.0 | 1/2 | 2 |
| vote | uniform | 60.4 | 0.0 | -0.2 | 1/2 | 2 |
| logodds>drives | uniform | 60.3 | 0.5 | -0.3 | 1/2 | 2 |
| median | uniform | 60.1 | 0.4 | -0.5 | 0/2 | 2 |

### Stacking over runs (common windows; stacker fitted out of sample)

| member | window macro-F1 | drive macro-F1 (mean) | drive macro-F1 (log-odds) | windows |
|---|---|---|---|---|
| pooled__size__gru__S0__s0 | 59.3 | 66.8 | 66.9 | 81440 |
| pooled__size__resnet__S0__s0 | 60.2 | 68.2 | 68.2 | 81440 |
| ensemble_mean | 61.8 | 68.8 | 68.7 | 81440 |
| stack_crossfit | 60.5 | 66.5 | 66.1 | 81440 |

## Powertrain: BEV vs ICE (split_platform) - internal folds (out-of-fold)

Runs: 2 (gru, resnet); kinds: oof_pooled; recipes: B0; macro-F1 in percent, mean over runs; `delta` = mean paired difference to `mean / uniform` (the protocol rule: mean over the drive's windows at the drive level, mean over the vehicle's pooled windows at the vehicle levels), `improved` = runs with a positive difference; `mean>drives / uniform` is the plan's two-stage vehicle candidate (log-odds mean of the drive decisions, n_windows^0.5 weights).

### Reference rule vs best rule per level and protocol

| level | protocol | mean / uniform (protocol) | mean>drives / uniform | best rule | macro-F1 | delta | improved | units |
|---|---|---|---|---|---|---|---|---|
| drive | full | 93.0 | - | mean / das0.5_ep0.5_steer0.5 | 94.8 | +1.8 | 2/2 | 2441 |
| vehicle | full | 93.4 | 94.1 | logodds / das1_ep0_steer0 | 95.5 | +2.1 | 2/2 | 467 |
| vehicle_multi | full | 93.6 | 94.8 | logodds / das1_ep0_steer0 | 96.2 | +2.6 | 2/2 | 275 |
| drive | human_only | 92.0 | - | gate_logodds / gate | 92.2 | +0.1 | 2/2 | 1621 |
| vehicle | human_only | 94.2 | 94.2 | gate_logodds / gate | 94.5 | +0.3 | 2/2 | 406 |
| vehicle_multi | human_only | 94.5 | 94.5 | vote / uniform | 95.0 | +0.5 | 2/2 | 230 |

### Drive level, protocol full: ranked rules (top 10 plus every unweighted rule)

| method | weights | macro-F1 | std over runs | delta | improved | runs |
|---|---|---|---|---|---|---|
| mean | das0.5_ep0.5_steer0.5 | 94.8 | 0.3 | +1.8 | 2/2 | 2 |
| logodds | das0.5_ep1_steer0 | 94.8 | 0.2 | +1.8 | 2/2 | 2 |
| logodds | das0.5_ep0.5_steer0.5 | 94.8 | 0.3 | +1.8 | 2/2 | 2 |
| mean | das0.5_ep1_steer0 | 94.7 | 0.1 | +1.7 | 2/2 | 2 |
| logodds | das0.5_ep0_steer0.5 | 94.7 | 0.2 | +1.7 | 2/2 | 2 |
| logodds | das0.5_ep0.5_steer0 | 94.7 | 0.2 | +1.7 | 2/2 | 2 |
| logodds | das1_ep0.5_steer0 | 94.7 | 0.1 | +1.7 | 2/2 | 2 |
| logodds | das0.5_ep1_steer0.5 | 94.7 | 0.3 | +1.7 | 2/2 | 2 |
| mean | das0.5_ep1_steer0.5 | 94.6 | 0.4 | +1.6 | 2/2 | 2 |
| logodds | das1_ep0_steer0 | 94.6 | 0.1 | +1.6 | 2/2 | 2 |
| gate_logodds | gate | 93.8 | 0.3 | +0.8 | 2/2 | 2 |
| logodds | uniform | 93.6 | 0.4 | +0.6 | 2/2 | 2 |
| gate_mean | gate | 93.4 | 0.3 | +0.4 | 2/2 | 2 |
| mean_T | uniform | 93.2 | 0.4 | +0.2 | 2/2 | 2 |
| mean | uniform | 93.0 | 0.5 | +0.0 | 0/2 | 2 |
| median | uniform | 91.7 | 0.2 | -1.3 | 0/2 | 2 |
| vote | uniform | 91.5 | 0.2 | -1.5 | 0/2 | 2 |

### Vehicle level, protocol full: ranked rules (top 10 plus every unweighted rule)

| method | weights | macro-F1 | std over runs | delta | improved | runs |
|---|---|---|---|---|---|---|
| logodds | das1_ep0_steer0 | 95.5 | 0.5 | +2.1 | 2/2 | 2 |
| logodds>drives | das0.5_ep0.5_steer0 | 95.5 | 0.0 | +2.1 | 2/2 | 2 |
| logodds>drives | das0.5_ep1_steer0 | 95.5 | 0.0 | +2.1 | 2/2 | 2 |
| logodds | das0_ep0.5_steer0.5 | 95.4 | 0.1 | +2.0 | 2/2 | 2 |
| logodds | das0.5_ep0.5_steer0.5 | 95.4 | 0.1 | +2.0 | 2/2 | 2 |
| mean>drives | das0.5_ep0_steer0 | 95.4 | 0.1 | +2.0 | 2/2 | 2 |
| logodds>drives | das0_ep0.5_steer0.5 | 95.4 | 0.1 | +2.0 | 2/2 | 2 |
| logodds>drives | das0_ep1_steer0.5 | 95.4 | 0.1 | +2.0 | 2/2 | 2 |
| mean | das0.5_ep0_steer0.5 | 95.4 | 0.1 | +2.0 | 2/2 | 2 |
| mean>drives | das0.5_ep0.5_steer0 | 95.4 | 0.1 | +2.0 | 2/2 | 2 |
| gate_logodds>drives | gate | 95.0 | 0.5 | +1.6 | 2/2 | 2 |
| gate_mean>drives | gate | 94.9 | 0.4 | +1.5 | 2/2 | 2 |
| logodds>drives | uniform | 94.9 | 0.6 | +1.5 | 2/2 | 2 |
| gate_logodds | gate | 94.7 | 1.0 | +1.4 | 2/2 | 2 |
| mean_T>drives | uniform | 94.5 | 0.8 | +1.1 | 2/2 | 2 |
| logodds | uniform | 94.5 | 1.0 | +1.1 | 2/2 | 2 |
| gate_mean | gate | 94.2 | 1.0 | +0.9 | 2/2 | 2 |
| mean>drives | uniform | 94.1 | 0.6 | +0.8 | 2/2 | 2 |
| mean_T | uniform | 94.1 | 1.1 | +0.8 | 2/2 | 2 |
| mean | uniform | 93.4 | 0.6 | +0.0 | 0/2 | 2 |
| median>drives | uniform | 92.6 | 0.9 | -0.8 | 0/2 | 2 |
| vote>drives | uniform | 92.3 | 0.5 | -1.1 | 0/2 | 2 |
| median | uniform | 91.7 | 0.5 | -1.7 | 0/2 | 2 |
| vote | uniform | 91.3 | 0.5 | -2.1 | 0/2 | 2 |

### Stacking over runs (common windows; stacker fitted out of sample)

| member | window macro-F1 | drive macro-F1 (mean) | drive macro-F1 (log-odds) | windows |
|---|---|---|---|---|
| pooled__bev_ice__gru__B0__s0 | 78.2 | 92.5 | 93.2 | 65285 |
| pooled__bev_ice__resnet__B0__s0 | 77.0 | 93.5 | 94.0 | 65285 |
| ensemble_mean | 79.4 | 93.9 | 94.2 | 65285 |
| stack_crossfit | 79.8 | 92.3 | 92.4 | 65285 |
