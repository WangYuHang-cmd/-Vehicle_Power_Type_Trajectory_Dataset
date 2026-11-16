# Source data of teaser A, B and C

Every number, line, bar and dot in `teaser_A.pptx`, `teaser_B.pptx` and `teaser_C.pptx` can be re-drawn from the files below. They are written by `demo/teaser/make_teaser_v2.py` together with the sub-figures (copies of the two build scripts are in `code/`).

## Where the data come from

- **Released dataset** (VTT release v1, public): drive tables at 10 Hz and the index files `drives`, `windows_60s` (teaser C, panel (b)-(e), car-model dots, region buckets).
- **Six held-out cases** chosen by `demo/teaser/make_selection.py` (`selection.json`): each is a car model never seen in training (split_platform test); the 60-s window and its drive are right on both tasks.
- **Model outputs**: final models of the paper, mean of seeds 0-2 (size: bi-GRU S9; powertrain: 1-D ResNet B5), recomputed and checked by `make_components.py` (`cases/*/meta.json`).
- **Cues**: definitions and episode choices in `selection.json`; per-drive shares in `cue_a_liftoff_per_drive.csv`; steering fits recomputed from the drive tables and checked against `selection.json`.
- **Badges**: numbers copied from `paper/OUTLINE.md` (line numbers in the CSV) and re-computed from the release index.
- **Map outline**: Natural Earth 1:110m (public domain), copied here.

## Privacy

Only the public pseudonyms of the release appear (`vehicle_####`, `drive_######`) plus car model names. Time is relative to the 60-s window or to the start of an episode; there are no positions, dates, clock times, device or drive identifiers of the raw logs, and region is the release's continent-level bucket.

## Units and conventions

speed km/h; acceleration m/s^2; steering-wheel angle deg (left positive); yaw rate deg/s (left positive); curvature 1/m; footprint = wheelbase x mean track in m^2 (SMALL < 4.43 <= MID < 4.88 <= LARGE); 0/1 flags for brake pedal, DAS and plotted points; probabilities 0-1 (printed values are rounded once from full precision).

## Files

| figure | file | rows | what |
|---|---|---|---|
| A | `teaser_A/A_cases.csv` | 6 | the six held-out cases of the 2 x 3 grid: labels, specs, public pseudonyms, car picture and its drawn width (car body width = scale x wheelbase) |
| A | `teaser_A/A_size_classes.csv` | 3 | column headers: footprint (wheelbase x mean track) ranges of the three size classes |
| A | `teaser_A/A_signals_LARGE-BEV_ford_f_150_lightning.csv` | 600 | signal panel (Ford F-150 Lightning, slide 1): the 600 samples (10 Hz) of the 60-s window |
| A | `teaser_A/A_signals_MID-BEV_genesis_gv60.csv` | 600 | signal panel (Genesis GV60, slide 2): the 600 samples (10 Hz) of the 60-s window |
| A | `teaser_A/A_model_outputs.csv` | 10 | decision bars: probability of each class for this window (dark bar, printed) and the mean over the drive (light bar) |
| A | `teaser_A/A_badges.csv` | 2 | the two result badges, copied from the paper outline |
| B | `teaser_B/B_cases.csv` | 3 | the three rows: labels, specs, public pseudonyms, car picture and drawn width |
| B | `teaser_B/B_frames.csv` | 9 | the three forward-camera stills per row: label and time in the window (= the dashed markers 1-3 on the signal lanes) |
| B | `teaser_B/B_signals_SMALL-BEV_hyundai_ioniq.csv` | 600 | signal lanes of the Hyundai Ioniq row: the 600 samples (10 Hz) of the 60-s window; das_any = 1 marks the grey DAS band |
| B | `teaser_B/B_signals_MID-ICE_genesis_gv70.csv` | 600 | signal lanes of the Genesis GV70 row: the 600 samples (10 Hz) of the 60-s window; das_any = 1 marks the grey DAS band |
| B | `teaser_B/B_signals_LARGE-BEV_ford_f_150_lightning.csv` | 600 | signal lanes of the Ford F-150 Lightning row: the 600 samples (10 Hz) of the 60-s window; das_any = 1 marks the grey DAS band |
| B | `teaser_B/B_model_outputs.csv` | 15 | model outputs of the three windows (bars and printed values) |
| B | `teaser_B/B_cue_a_episodes.csv` | 217 | cue (a): acceleration and brake pedal over the two deceleration episodes (tinted span = in_episode; the line is drawn where accel_drawn = 1) |
| B | `teaser_B/B_cue_a_episode_summary.csv` | 2 | cue (a): one row per episode |
| B | `teaser_B/B_cue_b_share_per_drive.csv` | 557 | cue (b): one row per held-out test drive (dots) |
| B | `teaser_B/B_cue_b_summary.csv` | 2 | cue (b): box (q25-q75) and median bar per powertrain |
| B | `teaser_B/B_cue_c_steering_samples.csv` | 2665 | cue (c): every sample of the two drives used in the fit (plotted = 1 for the up-to-1,200 dots drawn per car) |
| B | `teaser_B/B_cue_c_fits.csv` | 2 | cue (c): least-squares line through the origin per car, and the two ratios printed |
| C | `teaser_C/C_badges.csv` | 12 | badge row |
| C | `teaser_C/C_panel_a_car_models.csv` | 202 | panel (a): one dot per (car model, powertrain group: BEV / ICE / other = HEV, PHEV, mixed or unknown) at the median footprint of its drives |
| C | `teaser_C/C_panel_a_models_without_specs.csv` | 11 | panel (a): the car models without size specs (not drawn) |
| C | `teaser_C/C_panel_b_regions.csv` | 4 | panel (b): region buckets of the release (no finer location exists in the release) |
| C | `teaser_C/C_panel_c_hours_size_powertrain.csv` | 19 | panel (c): hours of all drives per size class x powertrain (the 'no size label' rows are not drawn) |
| C | `teaser_C/C_panel_d_contributors.csv` | 6 | panel (d): identity groups by number of drives (bars) and their hours |
| C | `teaser_C/C_panel_e_brands.csv` | 38 | panel (e): all brands; the top 8 have their own bar, the rest are summed |
| C | `teaser_C/ne_110m_admin0_countries.geojson` |  | panel (b): world outline, Natural Earth 1:110m admin-0 countries (public domain); countries are filled by region bucket and drawn without borders |

## Rebuild

```bash
source env.sh                                  # repository root
$CPY demo/teaser/make_teaser_v2.py             # sub-figures, layout.json, PARTS.md, this source_data folder
$PY  demo/teaser/build_teaser_pptx.py --preview  # teaser_A / B / C.pptx and previews
```
