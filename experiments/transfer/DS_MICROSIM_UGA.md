# Adapter `ds_microsim_uga.py`: microSIM-ACC / EV and the UGA EV platoon

Written 9 Nov 2026. Code: `experiments/transfer/ds_microsim_uga.py` (contract: `common.py`, `CONTRACT.md`). Two GNSS-only
car-following sets, delivered as two window sets:

| dataset | work dir | windows | vehicles | drives | hours | labels present |
|---|---|---|---|---|---|---|
| `microsim_ev` | `<data-root>/external/microsim_ev/work/` | 111 | 3 | 110 | 1.85 | BEV only; SMALL 8 / MID 103 |
| `uga_ev` | `<data-root>/external/uga_ev/work/` | 326 | 4 | 48 | 5.43 | BEV 244 / ICE 82; SMALL 243 / MID 83 |

Run: `source env.sh && $CPY experiments/transfer/ds_microsim_uga.py` (both sets, ~50 s; `--datasets`, `--dry-run`,
`--limit N`, `--no-length-correction`). Each work dir gets `windows.npz` + `meta.parquet` + `windows_summary.json`
(`common.save_windows`) and `adapter_report.json` (build statistics, per-trial / per-run QC, validation ranges, UGA clock
gates). Both sets are longitudinal-only: score them with the V2 variant (`score.py --dataset <name> --ckpts
'transfer__bev_ice__resnet__V2__s*,transfer__size__gru__V2__s*'`) or with the full models under `--mask-channels V2`.

## 1. Sources

* **microSIM-ACC / EV** (FAU + UC Berkeley; `external/microsim_ev/raw/unzipped/`): 294 trial CSVs, 50 Hz grid interpolated by
  the authors from a 25 Hz RaceBox GNSS logger; columns `Time` (s), `Speed Follower`, `Speed Leader` (km/h), their smoothed
  copies, `Spacing` (m). Folder levels = ACC gap setting (`Short / Medium / Long / XLong`), follower set-speed offset
  (`0 / 5 / 10_desired` mph above the free-flow speed), free-flow speed (`35 / 45 / 55 / 60` mph) and the speed the lead dips
  to (`0 / 15 / 25 / 35 / 45` mph), then the trial number (dissertation ch. 3, Figure 3.13). The follower is an ACC-engaged
  EV from the start of every trial (ACC is activated at free-flow speed before the recorded dip; stop-and-go trials keep it
  engaged through the stop); the lead is human-driven: a 2021 Toyota Camry for the Ioniq 5 trials, a 2022 Toyota Corolla
  sedan for the Polestar 2 and Tesla Model 3 trials.
* **UGA Mobility Lab EV platoon** (`external/uga_ev/raw/rawData/`): 48 CSVs = 12 runs x {LEADCAR, FLWCAR1, FLWCAR2, FLWCAR3},
  RaceBox 10 Hz GNSS (`Time` ISO-8601 UTC, lat / lon / altitude, `Speed` km/h, device-frame g-force, `Lap` always 0), all on
  28 Feb 2025, 6-8.4 min each, no gaps, no NaN. README (copy in `scratch/specs_ext_verify/readmes/uga_README.md`): one lead
  2021 Lexus UX 200 (ICE, ACC-equipped) followed by a 2024 Ford Mustang Mach-E, a 2023 Kia Niro EV and a second 2023 Kia
  Niro EV, on a straight low-traffic segment of GA-72; runs 1-8 have the followers on ACC (1-4 longest headway, 5-8 shortest),
  runs 9-12 manual; odd runs one-pedal driving (OPD), even runs two-pedal (TPD) by pairs (see the table in the code); the
  lead drives a scripted 0 -> 55 mph -> oscillations -> 35 mph -> oscillations -> stop profile with a low / high adaptation rate.

## 2. Channel mapping (both sets unless stated)

| channel | source | status |
|---|---|---|
| `v_mps` | GNSS speed / 3.6 (microSIM: raw `Speed Follower`, 50 Hz -> 10 Hz by linear resampling onto the 0.1-s grid, which coincides with every fifth source sample; UGA: `Speed` onto its own 10 Hz grid) | measured (GNSS Doppler, noise 0.1-0.3 m/s; training uses the filtered CAN speed) |
| `a_mps2` | `common.smooth_derivative(v)` (central difference at 10 Hz, 0.5-s moving average) | derived; sample-to-sample noise std(diff a) 0.086 (microSIM) / 0.080 (UGA) vs 0.135-0.166 on the training aEgo, so not noisier than training |
| `standstill` | `v < 0.1 m/s` | derived; the rule reproduces `carState.standstill` on 99.8 % of the training samples (3,000-window check) |
| `das_enabled` | microSIM: 1 everywhere (ACC follower by protocol). UGA: 1 for the three followers in runs 1-8, 0 in runs 9-12 (README table), 0 for the lead car | protocol flag, not a measured controller state |
| `lead_valid` | microSIM: 1 wherever the corrected gap is in (0.5, 165] m (99.95 % of Ioniq samples, 100 % Polestar / Tesla). UGA followers: 1 when the car ahead in the README order is ahead along the heading within (0.5, 165] m and less than 5 m to the side, AND the pair's GNSS clocks pass the consistency gate of section 4; UGA lead car: 0 | derived |
| `lead_d_m` | microSIM: `Spacing` minus the lead car's length (Camry 4.885 m, Corolla sedan 4.630 m). UGA: front-to-front distance between the two cars' positions (predecessor interpolated onto the follower's grid, local metres, projected on the follower's heading) minus the predecessor's length (UX 200 4.495, Mach-E 4.713, Niro EV 4.420 m) | derived (bumper-to-bumper gap assuming the same antenna offset in every car; `--no-length-correction` keeps the front-to-front distance) |
| `lead_v_rel_mps` | lead speed minus ego speed (both GNSS) | measured |
| `lead_speed_mps`, `thw_s`, `ttc_s`, `lead_a_slope_mps2` | `common.lead_channels` (lab rules: v + v_rel; d / v while v > 0.5 m/s; d / closing while closing > 0.1 m/s; causal 1-s slope of the lead speed inside lead episodes, clipped +-4) | derived with the training definitions |
| `steer_angle_deg`, `steer_rate_degps`, `steer_torque_norm`, `yaw_rate_radps`, `curv_1pm`, `lat_acc_mps2`, `steer_pressed`, `gas_pressed`, `brake_pressed` | not recorded (speed / position loggers only; the UGA g-force is in an unlevelled device frame and was not used) | missing (NaN everywhere) = the V2 longitudinal-only mask |

Window rule as in the lab: `cut_windows(..., align="run")`, 600 consecutive 10 Hz samples without a gap > 0.3 s, >= 20 %
moving samples; one trial (microSIM) or one file (UGA) is one `drive_id`; the trailing partial block is dropped.

## 3. Labels (from `specs_external.csv`, selected by model name + dataset tag)

| model | vehicle_id | footprint m^2 | size class | borderline | powertrain | VTT platform | source row |
|---|---|---|---|---|---|---|---|
| Hyundai Ioniq 5 (NE, 2022) | `microSIM Hyundai Ioniq 5 2022` | 4.826 | MID | yes (lab value 0.054 below the LARGE cut; public 3,000-mm wheelbase would give 4.91 LARGE) | BEV | HYUNDAI_IONIQ_5 | lab table, kept per CANDIDATES.md |
| Polestar 2 (2023) | `microSIM Polestar 2 2023` | 4.380 | SMALL | yes (0.050 below the MID cut) | BEV | POLESTAR_2 | lab table |
| Tesla Model 3 (2022) | `microSIM Tesla Model 3 2022` | 4.571 | MID | no | BEV | TESLA_MODEL_3 | lab table |
| Lexus UX 200 (ZA10, 2021) | `UGA Lexus UX 200 2021 (platoon lead)` | 4.092 | SMALL | no | ICE | - | web (medium confidence) |
| Ford Mustang Mach-E (Mk1, 2024) | `UGA Ford Mustang Mach-E 2024 (platoon 2nd)` | 4.845 | MID | yes (0.035 below the LARGE cut) | BEV | FORD_MUSTANG_MACH_E_MK1 | lab table |
| Kia Niro EV (SG2, 2023), two cars | `UGA Kia Niro EV 2023 (platoon 3rd)`, `... (platoon 4th)` | 4.255 | SMALL | no | BEV | KIA_NIRO_EV_2ND_GEN | lab table |

The UGA folder -> car assignment (FLWCAR1 = Mach-E, FLWCAR2 / FLWCAR3 = the two Niro EVs) is inferred from the README's
position table (lead, 2nd, 3rd, 4th); the GNSS positions confirm the platoon order LEADCAR > FLWCAR1 > FLWCAR2 > FLWCAR3
in every run (section 4), but nothing in the files names the cars. The two Niro EVs are kept as separate `vehicle_id`s.

## 4. Counts and validation

### microsim_ev

| model | trials | trials >= 60 s | windows | drives | median / max trial s | v max m/s | gap range m | lead_valid |
|---|---|---|---|---|---|---|---|---|
| Hyundai Ioniq 5 | 272 | 98 | 99 | 98 | 54.5 / 145 | 27.2 | -1.0 .. 62.0 (negative in one stop phase of one trial -> lead_valid 0 there) | 99.95 % |
| Polestar 2 | 11 | 8 | 8 | 8 | 66 / 89 | 24.7 | 10.5 .. 36.1 | 100 % |
| Tesla Model 3 | 11 | 4 | 4 | 4 | 54 / 95 | 25.6 | 10.6 .. 61.5 | 100 % |

184 trials (62 %) are shorter than 60 s and contribute nothing (119 of them 45-60 s); trials are separate recordings and cannot
be concatenated. Over the 111 windows: speed 0-27.2 m/s (mean 16.8), moving share 0.955 (min 0.477 in a stop-and-go trial),
standstill 2.7 % of samples, `das_enabled` 100 %, acceleration -5.41 .. +4.59 m/s^2 (p01 -2.38, p99 1.46; the extremes are
real braking-to-stop and catch-up phases with smooth speed traces, no single-sample speed jumps > 1 m/s), `lead_d_m`
0.9-62 m (p01 2.7, p99 58.6), `lead_v_rel` -7.5 .. +4.4 m/s, `thw` 0.70-22.3 s (p01 0.89, p99 5.6). Follower speed lags the
leader's by 0.7 s (median of the cross-correlation peak over 294 trials; 100 % non-negative) and d(Spacing)/dt agrees with
the speed difference (median r = 0.97), so the two speed columns and the spacing are consistent. No NaN in `v, a,
standstill, das_enabled, lead_valid`; the six lead values are NaN exactly where `lead_valid = 0`; `thw` / `ttc` / slope NaN
only by their definitions (3.6 % / 56 % / 0.3 %).

### uga_ev

| car (folder) | model | files | windows | hours | v max m/s | das_enabled | lead kept (runs) |
|---|---|---|---|---|---|---|---|
| LEADCAR | Lexus UX 200 | 12 | 82 | 1.48 | 25.1 | 0 | - (no vehicle ahead) |
| FLWCAR1 | Ford Mustang Mach-E | 12 | 83 | 1.46 | 27.1 | 1 in runs 1-8 | 11 of 12 (run 5 dropped) |
| FLWCAR2 | Kia Niro EV (3rd) | 12 | 82 | 1.46 | 29.0 | 1 in runs 1-8 | 12 of 12 |
| FLWCAR3 | Kia Niro EV (4th) | 12 | 79 | 1.43 | 29.3 | 1 in runs 1-8 | 8 of 12 (runs 5, 10, 11, 12 dropped) |

Every file is one gap-free 10 Hz recording (dt 0.1 s; one lead file has 0.08-0.12 s jitter, still no gap), so 6-8 windows per
file, none dropped for gaps or standstill. Over the 326 windows: speed 0-29.3 m/s (mean 18.5), moving share 0.9975 (min 0.863),
standstill 0.09 % of samples, `das_enabled` 50.3 % of samples (67 % on the followers, 0 on the lead), `lead_valid` 63.2 % of
samples and present in 210 windows (the 82 lead-car windows and 34 windows of the five clock-gated drives have none),
acceleration -7.52 .. +2.97 m/s^2 (p01 -1.43, p99 1.17; the two values below -5 are one braking-to-stop event of the lead and
the Mach-E in run 5; three samples in the whole set have a speed step > 1 m/s between consecutive samples), `lead_d_m`
0.5-111 m (p01 8.4, p99 76.8), `thw` p01 0.75 s, p99 3.5 s. Same NaN discipline as above.

**GNSS clock-consistency gate.** The lead gap comes from pairing two loggers by UTC time, so a clock offset between two
devices shifts the apparent gap by v x offset. A geometric test catches this: the time between the car ahead and the car
behind passing the same road point (front-to-front time gap, evaluated every 20 m along the lead's path) must stay well above
zero because the car ahead is 4.4-4.9 m long. 31 of the 36 follower drives pass (gap minimum >= 0.3 s and 10th percentile
>= 0.5 s; their medians are 2.2-2.8 s in the longest-headway runs 1-4, 1.2-1.5 s in the shortest-headway runs 6-8, 1.4-2.3 s
in the manual runs); five fail with apparent front-to-front gaps of -0.3 .. 0.6 s (run 5 FLWCAR1 and FLWCAR3; FLWCAR3 in runs
10, 11, 12, where it appears on top of FLWCAR2 for minutes), i.e. one logger's clock is off by roughly 1-2 s. Those five
drives keep their speed / acceleration windows but carry `lead_valid = 0` (the lead group is dropped exactly as the training
augmentation drops it); the note of every window says so. Median bumper-to-bumper gaps of the kept drives: 36.7 m (longest
ACC headway), 22.7 m (shortest), 32.5 m (manual). Smaller offsets (< 0.5 s) would pass the gate undetected and bias a gap by
up to ~10 m at highway speed; see the caveats.

### Scorer smoke test (consumption check, preliminary; the campaign's scoring step owns the final numbers)

`score.py` ran on both sets with the V2 longitudinal-only variants (`transfer__*__V2__s0, s1`) and with the full final models
under `--mask-channels V2` (outputs `score__*.md/json`, `probs__*.npz` in the work dirs). Vehicle-level decisions (mean
probability over all windows of the car -> arg-max), 200 vehicle-bootstrap resamples (uninformative with 1-2 vehicles per model):

| set | model (label) | V2 variant BEV/ICE: P(true) | V2 variant size | full + V2 mask BEV/ICE: P(true) | full + V2 mask size |
|---|---|---|---|---|---|
| microsim_ev | Hyundai Ioniq 5 (BEV, MID*) | 0.544 correct | SMALL wrong | 0.595 correct | SMALL wrong |
| microsim_ev | Polestar 2 (BEV, SMALL*) | 0.924 correct | MID wrong | 0.599 correct | LARGE wrong |
| microsim_ev | Tesla Model 3 (BEV, MID) | 0.541 correct | SMALL wrong | 0.315 wrong | SMALL wrong |
| uga_ev | Ford Mustang Mach-E (BEV, MID*) | 0.358 wrong | MID correct | 0.265 wrong | SMALL wrong |
| uga_ev | Kia Niro EV x2 (BEV, SMALL) | 0.553 correct (2/2) | MID wrong (0/2) | 0.351 wrong (0/2) | SMALL correct (2/2) |
| uga_ev | Lexus UX 200 (ICE, SMALL) | 0.422 wrong | SMALL correct | 0.626 correct | SMALL correct (0.81) |

`*` borderline footprint. Window-level accuracies are in the score files. Read these as a pipeline check only: the V2 size
variant is at chance on the internal test as well (VARIANTS.md), one physical car per model gives no interval, every
microSIM window is ACC-driven and every UGA BEV window comes from scripted platoon runs.

## 5. Caveats for the scoring step

1. **Protocol, not traffic.** Both sets are controlled car-following experiments: scripted speed dips (microSIM) or scripted
   lead oscillations (UGA) on closed-ish straight roads, ACC controllers doing the driving in all microSIM windows and in two
   thirds of the UGA follower windows. Compare with the lab's automation-engaged windows; the human-only protocol does not apply.
2. **One car per model** (two for the Niro EV), so the model level is the vehicle level and the bootstrap has no clusters to
   resample; report per model, and treat the Ioniq 5 and Mach-E size results as borderline (4.83 / 4.85 vs the 4.88 cut) and the
   Polestar 2 as borderline on the SMALL / MID cut (4.38 vs 4.43).
3. **Longitudinal-only input.** Nine of the 20 channels (all lateral channels, steer_pressed, both pedal flags) are missing;
   only the V2 variant is in distribution. The full model with `--mask-channels V2` is a secondary number.
4. **Speed and acceleration are GNSS-derived**, not CAN: Doppler speed with 0.1-0.3 m/s noise (microSIM's 50 Hz grid is an
   interpolation of 25 Hz data), acceleration a smoothed derivative (smoother than the training aEgo sample-to-sample, but
   lagging by ~0.25 s). The microSIM raw speed column was used; the authors' smoothed column is a causal filter that lags it.
5. **Lead distance convention.** microSIM's `Spacing` is the Haversine distance between the two GNSS antennas (the paper states
   no length correction, and the stopped-platoon spacings of 6-14 m, median 7.5 m, fit antenna-to-antenna with a 4.9-m Camry
   better than bumper-to-bumper); the adapter subtracts the lead car's public length, and does the same for UGA. If the
   authors had already removed the vehicle length, every microSIM gap is 4.6-4.9 m too short (thw about 0.2 s too short at
   highway speed); `--no-length-correction` rebuilds without the subtraction. The antenna offsets from the front bumpers are
   assumed equal across cars (windshield mounts), uncertainty about +-0.5 m.
6. **UGA GNSS clock offsets** (section 4): five follower drives lost their lead group; offsets below ~0.5 s in the kept drives
   cannot be excluded and would bias their gaps by up to ~10 m. A test using the lead channels on UGA should be compared with the
   same run under `--mask-channels lead`.
7. **UGA flags are protocol flags.** `das_enabled` follows the README table for the whole run (the manual start-up before
   ACC engagement and any take-over are not visible); the lead car's flag is set to 0 although the README calls it ACC-equipped
   (its scripted oscillations are assumed driver-executed). OPD vs TPD (one- / two-pedal regeneration) is recorded in the note
   of every window for a later regen probe but has no channel.
8. **microSIM coverage is thin outside the Ioniq 5**: 99 of 111 windows are one Ioniq 5; 8 Polestar 2 and 4 Tesla Model 3
   windows come from short-gap trials only. 62 % of the trials are shorter than one window.
9. **UGA follower identities are inferred** (README position order = folder order; the two Niro EVs are indistinguishable
   anyway). A wrong inference would only swap the Mach-E with a Niro EV.
10. No device or route identifiers exist in either source; vehicle and drive ids are public car names, folder paths and run numbers.
