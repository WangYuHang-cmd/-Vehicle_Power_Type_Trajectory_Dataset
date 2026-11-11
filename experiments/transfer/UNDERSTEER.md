# Understeer gradient of the steering-derived yaw (`understeer_fit.py`)

Written 11 Nov 2026. Companion of CCS_PIPELINE.md, section "Understeer-corrected yaw"; numbers from
`experiments/transfer/understeer_fit.json` (the repository copy of `<data-root>/tmp/understeer/fit.json`).

Sections: 1 Purpose, 2 Method, 3 Global k and its spread, 4 Why one global k, 5 What the scale factor a says,
6 k against footprint and class, 7 Per-platform table, 8 comma2k19 validation, 9 Caveats, 10 Reproduce.

## 1 Purpose

The external sets (`ccs`, `ccs_honda`, `comma2k19`) have no IMU / pose yaw; `ccs_decode.py` replaces it by the kinematic
bicycle model, yaw_b = v tan(delta_road) / L with delta_road = steering-wheel angle / steerRatio and L the wheelbase. That model
has no understeer. On comma2k19, whose 2018 logs carry a `carState.yawRate`, the ratio derived / logged grows with speed like
1 + k v^2: about 1.0 at 1-5 m/s, 1.5 at 20-25 m/s and 2.0-2.1 at 30-40 m/s for both cars (CCS_PIPELINE.md, gap 9). Yaw, curvature and
lateral acceleration of the external highway windows are therefore 1.5-2x too large relative to the lab's IMU / pose yaw the
models were trained on. The linear single-track steady state is yaw = v tan(delta_road) / (L (1 + k v^2)), so one k per vehicle
removes the bias; this study fits k on the lab's own released drive tables (where the measured yaw is the truth) and reduces it to
ONE global k for the decoder option `--yaw derived-understeer --understeer-k K`. The three re-decodes (`ccs_us`, `ccs_honda_us`,
`comma2k19_us`) use k = 0.00079289 s^2/m^2.

## 2 Method (as implemented in `understeer_fit.py`)

**Data.** Every drive table of the lab index with a platform and a steerRatio: 7,472 drives read, 6,657 usable. steerRatio is the
opendbc 0.3.1 CarSpecs value (the one the decoder takes from the logged / constructed carParams), else the 2024-08 opendbc
checkout, else the base platform of the lab name with generation / year suffixes stripped (column `sr_src`); L is the 0.3.1
wheelbase when the platform is in 0.3.1, else the lab spec. 361 lab platforms, 232 with a steerRatio, 225 with usable samples.

**Per-drive filters** (10 Hz rows): finite v, angle and yaw; `yaw_valid`; not standstill; v >= 2 m/s; |delta_road| >= 0.3 deg;
|yaw_meas| < 1.5 rad/s. A drive is skipped when it has fewer than 300 straight-driving samples for the angle offset
(v >= 5 m/s, |yaw_meas| < 0.02 rad/s; 700 drives, `no_straight`), when the correlation of steering angle and measured yaw at
v >= 2 m/s is below 0.5 (99 drives, `low_corr`), when that correlation is negative (2 drives, `neg_corr`, a sign error of one of the
two signals) or when fewer than 10 samples survive (14 drives, `no_samples`).

**Angle offset.** Several platforms log a constant steering-angle offset of a few degrees, which a no-intercept fit cannot absorb.
The primary variant (`offset`) subtracts, per drive, the median raw angle during straight driving; the `raw` variant keeps the
logged angle, which is what the decoder sees on the external sets. Both are fitted from one pass; `raw` is the sensitivity.

**Per platform** (eligible = at least 2 vehicles with usable samples, at least 1 h of usable samples, at least 1,000 samples):

    yaw_b   = v tan(delta_road) / L
    1-par:    yaw_b = (1 + k1 v^2) yaw_meas            least squares of (yaw_b - yaw_meas) on v^2 yaw_meas, no intercept
    2-par:    yaw_b = a (1 + k v^2) yaw_meas           linear in (a, a k); a absorbs a steerRatio / wheelbase error
    bins:     median of yaw_b / yaw_meas per speed bin (1-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30-40 m/s; |yaw_meas| >= 0.01 rad/s)
    k_bin:    ratio - 1 = k v^2 through the bin medians (bin centre, weighted by samples), a cross-check of the sample fits

Each fit has one robust pass: residual outliers beyond 3 scaled MADs of the median residual are dropped and the fit repeated;
R^2 and the kept fraction refer to the final fit. The 1-par k1 is the decoder's form and is what the global k is made of; the
2-par k is the understeer gradient with the geometry error factored out.

**Global k** = hours-weighted median of k1 over the eligible platforms whose modal `split_platform` is train or validation
(held-out test and excluded platforms are fitted and tabulated but do not enter the global value). Also reported: plain median,
interquartile range, hours-weighted mean, the same for the 2-par k and for k_bin, and the position of a.

**Subsets.** The primary numbers use all samples, because the external features are decoded from all samples too; the fit is
repeated on the human-only subset (`op_lat_active` false) and on the lateral-controller-on subset, for both angle variants.

**comma2k19 validation.** From the verifier's per-segment logged carState dumps (100 Hz): the corrected derived yaw
yaw_b / (1 + k v^2) with the constructed carParams (RAV4 steerRatio 16.88 / L 2.65 m, Civic 15.38 / 2.70 m) against the logged
yawRate, binned median ratio per speed bin, on 10 evenly spaced segments per chunk, on all segments, and on the human-only
samples of all segments (`controlsState.enabled` false). Variants: `uncorrected`, `global_k`, `own_k` (the car's own lab k1,
when the platform is fitted in the lab), `model_2018_ak` (the (a, k) that 2018 openpilot's single-track vehicle model implies).

## 3 The global k and its spread

78 eligible platforms with 468.9 h of usable samples; 62 of them train / validation with 423.8 h.

| statistic (offset variant, all samples) | 1-par k1, train / val (n = 62) | 2-par k, train / val (n = 62) | k_bin, train / val (n = 62) | 1-par k1, all eligible (n = 78) |
|---|---|---|---|---|
| hours-weighted median | **0.000793** (= k_global) | 0.000752 | 0.000785 | 0.000807 |
| plain median | 0.000837 | 0.000855 | 0.000833 | 0.000939 |
| interquartile range | 0.000690 - 0.001223 | 0.000731 - 0.000985 | 0.000687 - 0.001104 | 0.000690 - 0.001274 |
| hours-weighted mean | 0.000890 | 0.000792 | 0.000840 | 0.000910 |
| min - max | 0.000201 - 0.002421 | 0.000338 - 0.001884 | 0.000376 - 0.002016 | -0.000155 - 0.002505 |

All in s^2/m^2. k_global = 0.00079289 s^2/m^2; its characteristic speed 1 / sqrt(k) = 35.5 m/s is where the correction factor
reaches 2, consistent with the comma2k19 ratio of about 2 at 30-40 m/s. The 2-par k is tighter (interquartile width 0.00025 against
0.00053) because the 1-par k1 also has to absorb the geometry scale a (section 5); the hours-weighted medians of the three
estimators agree within 5.2 %.

Sensitivity (same statistic, k_global per variant / subset): raw angle, all samples 0.000822 (62 platforms); offset, human-only
0.000815 (39 platforms, 151.4 h); offset, controller-on 0.000787 (44 platforms, 243.2 h); raw, human-only 0.000809; raw,
controller-on 0.000798. The choice of angle variant or sample subset moves the global value by at most 3.7 %.

## 4 Why one global k and not a per-platform k

The external features are the inputs of a zero-shot transfer test. A per-platform k fitted on the lab's IMU / pose yaw is a
platform-specific calibration taken from the training data, and applying it to an external decode would carry that calibration
into the external features (and would only exist for the platforms the lab has, i.e. mostly the seen ones). One global constant
removes the speed-dependent bias that every car shares, and leaves the platform-specific residual where the decoder's geometry
already is: in the external data. The lab's own k for a car is therefore reported (table, section 7, and the `own_k` rows of
section 8) but never used.

## 5 What the scale factor a says

The 2-par fit returns a = 1 when the logged carParams steerRatio and wheelbase are right. Over the 62 train / validation platforms
a has hours-weighted median 1.046, plain median 1.021, interquartile range 0.955 - 1.110 and range 0.827 - 1.236 (all 78 eligible:
median 1.046, range 0.795 - 1.236). a > 1.10 on 32.1 % of the eligible platforms and a < 0.90 on 10.3 %: for about a third of the
platforms the derived yaw is more than 10 % too large at every speed, and for a further tenth more than 10 % too small. This is a
geometry error (steerRatio, wheelbase, or a steering-angle scale) of the logged / constructed carParams, not understeer: it is
present at 1-5 m/s where the understeer term is below 2 % (e.g. 1.28 for the Model X and Acura RDX, 0.85 for the GV60 and
Stinger). A speed-dependent bias k cannot fix it, and the external decodes inherit it, since they use the same carParams
geometry. In the 1-par fit this scale is absorbed by k1 (platforms with a > 1 get a larger k1, those with a < 1 a smaller or
even negative one, e.g. the GV60's -0.000155), which is why the 1-par spread is wider than the 2-par spread and why the
hours-weighted median, not the mean, is used.

## 6 Does k relate to footprint or class

No. Spearman rank correlation of k with footprint: -0.11 (1-par) / -0.26 (2-par); of the 1-par k with wheelbase: -0.09. Median
k1 by size class: SMALL 0.001007 (35 platforms), MID 0.000867 (22), LARGE 0.000837 (21); by powertrain: ICE 0.000837, PHEV
0.000828, BEV 0.000957, HEV 0.001237. The class medians sit inside each other's interquartile ranges, so one global k does not
favour a class, and the correction carries no class information into the features (the yaw-masked check of the collector
confirms that the non-yaw channels are untouched).

## 7 Per-platform table

All 78 eligible platforms, offset variant, all samples, sorted by the 1-par k1 (k values in 1e-3 s^2/m^2; "hours used" and
"samples" are the usable samples after the filters; "kept" is the fraction kept by the robust pass of the 1-par fit; the
speed-bin columns are the median yaw_b / yaw_meas per bin, "-" = fewer than 50 samples). Split = modal `split_platform` of the
platform's drives; held-out test and excluded platforms do not enter k_global.

| platform | class | footprint m2 | split | vehicles | hours used | samples | k_1par (1e-3 s2/m2) | a | k_2par (1e-3) | k_bin (1e-3) | R2 (1-par / 2-par) | kept | steerRatio (src) | L m | median |offset| deg | median ratio yaw_b / yaw_meas per bin 1-5 / 5-10 / 10-15 / 15-20 / 20-25 / 25-30 / 30-40 m/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GENESIS_GV60_EV_1ST_GEN | MID | 4.73 | held-out test | 3 | 3.0 | 108804 | -0.155 | 0.816 | 1.084 | 0.339 | 0.975 / 0.992 | 0.899 | 17.60 (opendbc_0.3.1) | 2.90 | 0.2 | 0.85 / 0.91 / 1.03 / 1.09 / 1.16 / 1.41 / 1.63 |
| KIA_STINGER | MID | 4.51 | held-out test | 3 | 3.7 | 131889 | 0.149 | 0.795 | 1.210 | 0.301 | 0.985 / 0.993 | 0.896 | 16.56 (opendbc_0.3.1) | 2.78 | 0.4 | 0.85 / 0.91 / 1.00 / 1.10 / 1.25 / 1.54 / - |
| HONDA_ODYSSEY_5G_MMR | LARGE | 5.14 | train | 5 | 5.9 | 210920 | 0.201 | 0.850 | 0.924 | 0.376 | 0.978 / 0.994 | 0.927 | 19.40 (opendbc_0.3.1) | 3.00 | 1.2 | 0.86 / 0.91 / 1.02 / 1.10 / 1.19 / 1.41 / 1.49 |
| FORD_MUSTANG_MACH_E_MK1 | MID | 4.84 | train | 10 | 4.4 | 159435 | 0.372 | 0.865 | 0.793 | 0.585 | 0.971 / 0.986 | 0.886 | 17.00 (opendbc_0.3.1) | 2.98 | 0.3 | 0.89 / 0.91 / 0.96 / 1.08 / 1.18 / 1.30 / 1.92 |
| SUBARU_IMPREZA_2020 | SMALL | 4.13 | train | 11 | 1.3 | 48088 | 0.380 | 0.867 | 0.752 | 0.446 | 0.986 / 0.994 | 0.885 | 17.00 (opendbc_0.3.1) | 2.67 | 1.2 | 0.87 / 0.92 / 0.99 / 1.07 / 1.19 / 1.34 / 1.65 |
| ACURA_MDX_4G_MMR | LARGE | 4.97 | train | 2 | 5.3 | 190965 | 0.435 | 0.968 | 0.591 | 0.645 | 0.980 / 0.980 | 0.939 | 16.20 (opendbc_0.3.1) | 2.89 | 0.8 | 0.98 / 0.98 / 1.09 / 1.21 / 1.35 / 1.57 / 1.73 |
| SUBARU_FORESTER | SMALL | 4.19 | train | 9 | 2.9 | 104818 | 0.482 | 0.914 | 0.781 | 0.503 | 0.989 / 0.992 | 0.907 | 17.00 (opendbc_0.3.1) | 2.67 | 1.0 | 0.92 / 0.97 / 1.05 / 1.14 / 1.27 / 1.40 / 1.62 |
| CHEVROLET_VOLT | SMALL | 4.18 | train | 11 | 7.7 | 275632 | 0.484 | 1.007 | 0.442 | 0.413 | 0.994 / 0.994 | 0.883 | 17.70 (opendbc_0.3.1) | 2.69 | 0.2 | 1.03 / 1.05 / 1.10 / 1.12 / 1.17 / 1.27 / 1.54 |
| RAM_1500_5TH_GEN | LARGE | 6.73 | train | 16 | 3.9 | 140136 | 0.486 | 0.981 | 0.541 | 0.455 | 0.992 / 0.992 | 0.921 | 16.30 (opendbc_0.3.1) | 3.88 | 3.2 | 0.99 / 1.00 / 1.09 / 1.16 / 1.24 / 1.34 / 1.53 |
| PORSCHE_MACAN_MK1 | MID | 4.64 | excluded | 2 | 1.7 | 60633 | 0.493 | 0.944 | 0.726 | 0.551 | 0.990 / 0.991 | 0.932 | 16.20 (opendbc_0.3.1) | 2.81 | 0.4 | 0.96 / 0.98 / 1.08 / 1.17 / 1.28 / 1.45 / 1.70 |
| KIA_EV6 | MID | 4.74 | train | 29 | 10.4 | 374515 | 0.494 | 0.954 | 0.638 | 0.497 | 0.995 / 0.995 | 0.896 | 16.00 (opendbc_0.3.1) | 2.90 | 0.6 | 0.97 / 1.00 / 1.08 / 1.16 / 1.26 / 1.39 / 1.60 |
| HONDA_CIVIC_2022 | SMALL | 4.21 | train | 24 | 9.0 | 325582 | 0.497 | 0.926 | 0.827 | 0.473 | 0.991 / 0.993 | 0.883 | 15.38 (opendbc_0.3.1) | 2.70 | 0.2 | 0.94 / 0.96 / 1.06 / 1.17 / 1.25 / 1.40 / 1.56 |
| RIVIAN_R1_GEN1 | LARGE | 5.58 | train | 37 | 48.9 | 1759365 | 0.516 | 1.046 | 0.383 | 0.532 | 0.990 / 0.990 | 0.901 | 15.20 (opendbc_0.3.1:RIVIAN_R1) | 3.26 | 1.1 | 1.06 / 1.05 / 1.10 / 1.18 / 1.27 / 1.42 / 1.57 |
| HONDA_CIVIC_BOSCH | SMALL | 4.20 | train | 13 | 30.0 | 1078300 | 0.548 | 0.887 | 1.248 | 0.786 | 0.986 / 0.992 | 0.908 | 15.38 (opendbc_0.3.1) | 2.70 | 0.1 | 0.92 / 1.01 / 1.15 / 1.29 / 1.43 / 1.54 / 1.87 |
| HYUNDAI_PALISADE | LARGE | 4.96 | validation | 3 | 4.9 | 177522 | 0.592 | 0.927 | 0.892 | 0.643 | 0.993 / 0.995 | 0.900 | 17.94 (opendbc_0.3.1) | 2.90 | 1.0 | 0.95 / 0.98 / 1.08 / 1.18 / 1.31 / 1.58 / 1.66 |
| KIA_CEED_PHEV | SMALL | 4.16 | held-out test | 2 | 1.7 | 60078 | 0.625 | 0.966 | 0.698 | 0.539 | 0.995 / 0.996 | 0.943 | 13.75 (opendbc_0.3.1:KIA_CEED) | 2.65 | 0.5 | 0.98 / 1.00 / 1.03 / 1.13 / 1.29 / 1.45 / 1.66 |
| HONDA_CITY_7G | SMALL | 3.86 | train | 3 | 1.3 | 48487 | 0.626 | 0.827 | 1.250 | 0.694 | 0.978 / 0.990 | 0.924 | 19.00 (opendbc_0.3.1) | 2.60 | 0.5 | 0.84 / 0.93 / 1.00 / 1.11 / 1.34 / 1.62 / 2.10 |
| HYUNDAI_PALISADE_2023 | LARGE | 4.96 | train | 6 | 1.7 | 62312 | 0.626 | 0.920 | 1.050 | 0.712 | 0.991 / 0.992 | 0.898 | 17.94 (opendbc_0.3.1:HYUNDAI_PALISADE) | 2.90 | 0.6 | 0.93 / 0.99 / 1.10 / 1.18 / 1.43 / 1.58 / 1.95 |
| HONDA_ACCORD_11G | MID | 4.56 | train | 8 | 5.4 | 196080 | 0.684 | 0.996 | 0.696 | 0.683 | 0.995 / 0.995 | 0.918 | 16.00 (opendbc_0.3.1) | 2.83 | 0.9 | 0.98 / 1.04 / 1.14 / 1.21 / 1.33 / 1.52 / 1.78 |
| HYUNDAI_IONIQ_6 | MID | 4.80 | train | 14 | 15.8 | 569746 | 0.689 | 1.059 | 0.481 | 0.605 | 0.993 / 0.995 | 0.891 | 14.26 (opendbc_0.3.1) | 2.97 | 0.9 | 1.08 / 1.10 / 1.18 / 1.23 / 1.30 / 1.43 / 1.66 |
| HONDA_PILOT_4G | LARGE | 4.97 | train | 9 | 1.5 | 55413 | 0.691 | 0.956 | 0.853 | 0.716 | 0.992 / 0.993 | 0.919 | 17.50 (opendbc_0.3.1) | 2.89 | 1.2 | 0.98 / 1.00 / 1.12 / 1.21 / 1.34 / 1.63 / 1.88 |
| FORD_ESCAPE_MK4_5 | SMALL | 4.39 | train | 3 | 7.5 | 270106 | 0.691 | 0.974 | 0.771 | 0.663 | 0.994 / 0.994 | 0.906 | 16.70 (opendbc_0.3.1) | 2.71 | 0.1 | 0.97 / 0.99 / 1.06 / 1.21 / 1.35 / 1.58 / 1.75 |
| HONDA_ACCORD | MID | 4.54 | train | 13 | 4.0 | 145734 | 0.720 | 0.899 | 0.987 | 0.750 | 0.988 / 0.990 | 0.905 | 16.33 (opendbc_0.3.1) | 2.83 | 0.7 | 0.92 / 1.00 / 1.15 / 1.28 / 1.40 / 1.56 / 1.90 |
| FORD_MAVERICK_MK1 | LARGE | 4.93 | train | 5 | 2.5 | 91311 | 0.746 | 0.973 | 0.815 | 0.747 | 0.991 / 0.991 | 0.919 | 17.00 (opendbc_0.3.1) | 3.08 | 0.6 | 1.00 / 1.01 / 1.07 / 1.23 / 1.36 / 1.58 / 1.92 |
| HONDA_CRV_6G | SMALL | 4.37 | train | 13 | 3.5 | 126988 | 0.748 | 0.922 | 1.103 | 0.894 | 0.991 / 0.992 | 0.903 | 16.20 (opendbc_0.3.1) | 2.70 | 0.2 | 0.96 / 0.98 / 1.11 / 1.21 / 1.39 / 1.68 / 2.40 |
| FORD_ESCAPE_MK4 | SMALL | 4.39 | validation | 5 | 1.5 | 55591 | 0.750 | 0.961 | 0.893 | 0.775 | 0.992 / 0.992 | 0.915 | 16.70 (opendbc_0.3.1) | 2.71 | 0.6 | 0.97 / 1.00 / 1.11 / 1.26 / 1.39 / 1.60 / 1.92 |
| SUBARU_IMPREZA | SMALL | 4.13 | train | 7 | 1.4 | 50475 | 0.760 | 0.976 | 0.871 | 0.860 | 0.994 / 0.994 | 0.897 | 15.00 (opendbc_0.3.1) | 2.67 | 1.3 | 1.01 / 1.03 / 1.12 / 1.26 / 1.36 / 1.74 / 2.53 |
| TESLA_MODEL_3 | MID | 4.57 | validation | 23 | 16.8 | 605156 | 0.772 | 1.017 | 0.683 | 0.726 | 0.994 / 0.994 | 0.916 | 12.00 (opendbc_0.3.1) | 2.88 | 0.4 | 1.04 / 1.09 / 1.16 / 1.25 / 1.35 / 1.48 / 1.74 |
| CHEVROLET_SILVERADO | LARGE | 6.65 | train | 8 | 14.5 | 521005 | 0.793 | 1.021 | 0.716 | 0.684 | 0.993 / 0.994 | 0.909 | 16.30 (opendbc_0.3.1) | 3.75 | 1.4 | 1.03 / 1.05 / 1.15 / 1.25 / 1.37 / 1.54 / 1.77 |
| TOYOTA_HIGHLANDER_TSS2 | MID | 4.68 | train | 11 | 1.1 | 40774 | 0.797 | 1.008 | 0.753 | 0.771 | 0.994 / 0.994 | 0.886 | 16.00 (opendbc_0.3.1) | 2.82 | 1.0 | 1.03 / 1.06 / 1.16 / 1.26 / 1.38 / 1.56 / 1.86 |
| HYUNDAI_SONATA_HYBRID | MID | 4.62 | train | 10 | 8.3 | 297234 | 0.805 | 0.936 | 1.037 | 0.799 | 0.994 / 0.995 | 0.899 | 15.26 (opendbc_0.3.1) | 2.84 | 3.1 | 0.95 / 0.99 / 1.12 / 1.25 / 1.42 / 1.63 / 1.93 |
| RIVIAN_R1 | LARGE | 5.58 | train | 17 | 25.7 | 926122 | 0.807 | 1.155 | 0.338 | 0.685 | 0.986 / 0.992 | 0.871 | 15.20 (opendbc_0.3.1) | 3.08 | 0.9 | 1.18 / 1.16 / 1.22 / 1.31 / 1.38 / 1.43 / 1.63 |
| TOYOTA_RAV4_PRIME_2021 | SMALL | 4.32 | validation | 3 | 1.3 | 47706 | 0.810 | 0.943 | 0.942 | 0.729 | 0.991 / 0.992 | 0.931 | 16.88 (opendbc_0.3.1:TOYOTA_RAV4_PRIME) | 2.69 | 0.9 | 0.95 / 0.99 / 1.12 / 1.26 / 1.42 / 1.64 / 1.85 |
| TOYOTA_RAV4_PRIME | SMALL | 4.32 | train | 13 | 1.5 | 53200 | 0.828 | 0.960 | 0.991 | 0.837 | 0.993 / 0.994 | 0.896 | 16.88 (opendbc_0.3.1) | 2.68 | 0.6 | 0.99 / 1.02 / 1.14 / 1.30 / 1.41 / 1.66 / 1.92 |
| FORD_EXPLORER_MK6 | LARGE | 5.14 | train | 7 | 1.7 | 62562 | 0.837 | 1.029 | 0.764 | 0.799 | 0.990 / 0.990 | 0.941 | 16.80 (opendbc_0.3.1) | 3.02 | 0.2 | 1.04 / 1.10 / 1.17 / 1.26 / 1.41 / 1.62 / 1.94 |
| TOYOTA_RAV4 | SMALL | 4.13 | train | 7 | 1.3 | 47284 | 0.837 | 0.969 | 0.976 | 0.891 | 0.994 / 0.994 | 0.899 | 16.88 (opendbc_0.3.1) | 2.65 | 2.2 | 0.97 / 1.03 / 1.15 / 1.26 / 1.41 / 1.78 / 2.06 |
| TOYOTA_PRIUS | SMALL | 4.13 | train | 31 | 11.4 | 409438 | 0.869 | 0.990 | 0.896 | 0.785 | 0.993 / 0.992 | 0.913 | 15.74 (opendbc_0.3.1) | 2.70 | 1.3 | 1.01 / 1.05 / 1.15 / 1.28 / 1.43 / 1.67 / 1.91 |
| TOYOTA_RAV4_TSS2_2022 | SMALL | 4.34 | train | 10 | 2.9 | 105653 | 0.904 | 1.127 | 0.623 | 0.826 | 0.984 / 0.991 | 0.926 | 14.30 (opendbc_0.3.1) | 2.69 | 1.8 | 1.13 / 1.17 / 1.29 / 1.38 / 1.44 / 1.61 / 1.96 |
| TESLA_MODEL_Y | MID | 4.73 | held-out test | 21 | 3.2 | 114405 | 0.929 | 1.043 | 0.745 | 0.818 | 0.993 / 0.994 | 0.908 | 12.00 (opendbc_0.3.1) | 2.89 | 0.4 | 1.07 / 1.11 / 1.22 / 1.31 / 1.43 / 1.54 / 1.85 |
| FORD_EXPEDITION_MK4 | LARGE | 5.53 | train | 2 | 1.0 | 36477 | 0.950 | 1.063 | 0.727 | 0.921 | 0.990 / 0.993 | 0.912 | 17.00 (opendbc_0.3.1) | 3.69 | 0.6 | 1.10 / 1.08 / 1.18 / 1.30 / 1.49 / 1.61 / 2.24 |
| GENESIS_GV70_1ST_GEN | MID | 4.73 | held-out test | 8 | 1.7 | 62084 | 0.967 | 1.061 | 0.763 | 0.847 | 0.984 / 0.987 | 0.948 | 14.60 (opendbc_0.3.1) | 2.87 | 0.5 | 1.08 / 1.14 / 1.28 / 1.42 / 1.67 / 1.69 / 1.81 |
| FORD_F_150_MK14 | LARGE | 6.44 | train | 15 | 17.1 | 615702 | 0.982 | 1.102 | 0.693 | 0.884 | 0.985 / 0.991 | 0.911 | 17.00 (opendbc_0.3.1) | 3.69 | 0.8 | 1.11 / 1.14 / 1.27 / 1.38 / 1.50 / 1.67 / 1.99 |
| KIA_NIRO_EV | SMALL | 4.27 | train | 12 | 10.0 | 359067 | 0.984 | 1.072 | 0.748 | 0.877 | 0.992 / 0.995 | 0.903 | 13.60 (opendbc_0.3.1) | 2.70 | 0.6 | 1.09 / 1.12 / 1.23 / 1.38 / 1.52 / 1.65 / 1.94 |
| FORD_F_150_LIGHTNING_MK1 | LARGE | 6.41 | held-out test | 3 | 1.6 | 56016 | 0.991 | 1.084 | 0.736 | 0.850 | 0.990 / 0.994 | 0.901 | 16.90 (opendbc_0.3.1) | 3.70 | 0.1 | 1.12 / 1.14 / 1.26 / 1.40 / 1.50 / 1.69 / 1.94 |
| HYUNDAI_IONIQ_5_PE | MID | 4.88 | train | 9 | 5.0 | 181300 | 1.002 | 1.069 | 0.687 | 0.829 | 0.991 / 0.995 | 0.906 | 14.26 (opendbc_0.3.1:HYUNDAI_IONIQ_5) | 3.00 | 0.6 | 1.10 / 1.12 / 1.21 / 1.31 / 1.42 / 1.58 / 1.86 |
| CHEVROLET_BOLT_EUV | SMALL | 3.99 | validation | 18 | 6.4 | 229146 | 1.007 | 1.022 | 0.938 | 0.999 | 0.992 / 0.993 | 0.921 | 16.80 (opendbc_0.3.1) | 2.64 | 0.4 | 1.04 / 1.09 / 1.24 / 1.37 / 1.50 / 1.77 / 2.15 |
| HYUNDAI_SONATA | MID | 4.62 | train | 11 | 2.5 | 90443 | 1.066 | 0.954 | 1.219 | 1.080 | 0.991 / 0.992 | 0.914 | 15.26 (opendbc_0.3.1) | 2.84 | 1.3 | 0.96 / 1.02 / 1.15 / 1.31 / 1.57 / 1.85 / 2.31 |
| HYUNDAI_IONIQ_5 | MID | 4.83 | train | 29 | 14.6 | 525312 | 1.102 | 1.106 | 0.699 | 0.924 | 0.988 / 0.994 | 0.913 | 14.26 (opendbc_0.3.1) | 2.97 | 0.7 | 1.12 / 1.16 / 1.27 / 1.37 / 1.48 / 1.66 / 1.95 |
| HONDA_CRV_5G | SMALL | 4.28 | train | 8 | 1.7 | 59565 | 1.115 | 0.933 | 1.481 | 1.125 | 0.990 / 0.991 | 0.911 | 16.00 (opendbc_0.3.1) | 2.66 | 0.3 | 0.96 / 1.07 / 1.26 / 1.36 / 1.57 / 1.76 / 2.20 |
| TOYOTA_RAV4_TSS2_2023 | SMALL | 4.34 | train | 13 | 4.6 | 166551 | 1.138 | 1.111 | 0.753 | 0.967 | 0.988 / 0.994 | 0.911 | 14.30 (opendbc_0.3.1) | 2.69 | 0.5 | 1.12 / 1.17 / 1.27 / 1.38 / 1.53 / 1.78 / 1.95 |
| TOYOTA_CAMRY_TSS2 | MID | 4.47 | train | 12 | 4.9 | 176894 | 1.167 | 1.151 | 0.653 | 0.924 | 0.984 / 0.995 | 0.910 | 13.70 (opendbc_0.3.1) | 2.82 | 0.9 | 1.16 / 1.19 / 1.30 / 1.40 / 1.50 / 1.70 / 1.91 |
| TOYOTA_RAV4_TSS2 | SMALL | 4.34 | train | 31 | 7.8 | 280570 | 1.206 | 1.094 | 0.857 | 1.055 | 0.989 / 0.992 | 0.925 | 14.30 (opendbc_0.3.1) | 2.69 | 0.7 | 1.09 / 1.14 / 1.27 / 1.40 / 1.55 / 1.82 / 2.09 |
| SUBARU_FORESTER_PREGLOBAL | SMALL | 4.14 | train | 2 | 1.7 | 60337 | 1.216 | 1.212 | 0.744 | 1.115 | 0.975 / 0.991 | 0.902 | 20.00 (opendbc_0.3.1) | 2.67 | 0.5 | 1.22 / 1.26 / 1.38 / 1.57 / 1.66 / 1.90 / 2.30 |
| SUBARU_ASCENT_2023 | MID | 4.72 | validation | 4 | 2.1 | 74292 | 1.226 | 1.141 | 0.764 | 1.057 | 0.987 / 0.995 | 0.906 | 13.50 (opendbc_0.3.1) | 2.89 | 1.4 | 1.13 / 1.18 / 1.34 / 1.45 / 1.55 / 1.79 / 1.94 |
| TOYOTA_COROLLA_TSS2 | SMALL | 4.09 | held-out test | 38 | 12.5 | 450596 | 1.237 | 1.120 | 0.817 | 1.077 | 0.986 / 0.993 | 0.902 | 13.90 (opendbc_0.3.1) | 2.67 | 1.4 | 1.14 / 1.18 / 1.29 / 1.39 / 1.58 / 1.78 / 2.20 |
| CHRYSLER_PACIFICA_2019_HYBRID | LARGE | 5.36 | excluded | 6 | 1.2 | 43255 | 1.248 | 1.073 | 0.803 | 1.039 | 0.991 / 0.994 | 0.869 | 16.20 (opendbc_0.3.1) | 3.09 | 1.0 | 1.07 / 1.12 / 1.28 / 1.39 / 1.52 / 1.82 / 2.03 |
| MAZDA_CX5_2022 | SMALL | 4.31 | validation | 11 | 1.5 | 53537 | 1.257 | 1.054 | 1.082 | 1.160 | 0.990 / 0.992 | 0.930 | 15.50 (opendbc_0.3.1) | 2.70 | 0.4 | 1.08 / 1.14 / 1.36 / 1.50 / 1.62 / 1.90 / 2.33 |
| KIA_NIRO_PHEV_2022 | SMALL | 4.29 | train | 4 | 5.9 | 212272 | 1.269 | 1.119 | 0.912 | 1.113 | 0.987 / 0.993 | 0.897 | 13.60 (opendbc_0.3.1) | 2.70 | 0.8 | 1.15 / 1.18 / 1.30 / 1.42 / 1.52 / 1.88 / 2.29 |
| HYUNDAI_KONA_EV | SMALL | 4.02 | held-out test | 9 | 2.1 | 77378 | 1.275 | 1.076 | 0.696 | 1.046 | 0.992 / 0.995 | 0.906 | 13.42 (opendbc_0.3.1) | 2.60 | 1.5 | 1.10 / 1.12 / 1.21 / 1.30 / 1.41 / 1.64 / - |
| KIA_SORENTO_HEV_4TH_GEN | MID | 4.64 | held-out test | 2 | 1.4 | 49606 | 1.294 | 1.113 | 0.836 | 1.094 | 0.987 / 0.992 | 0.908 | 13.50 (opendbc_0.3.1) | 2.81 | 1.9 | 1.12 / 1.15 / 1.22 / 1.40 / 1.36 / 1.85 / 2.17 |
| HYUNDAI_KONA_2022 | SMALL | 4.02 | held-out test | 3 | 1.0 | 36620 | 1.305 | 1.082 | 0.979 | 1.155 | 0.991 / 0.995 | 0.905 | 13.42 (opendbc_0.3.1) | 2.60 | 2.6 | 1.11 / 1.16 / 1.30 / 1.41 / 1.64 / 1.82 / 2.24 |
| CHEVROLET_VOLT_2019 | SMALL | 4.19 | train | 2 | 2.9 | 105384 | 1.318 | 1.153 | 0.902 | 1.285 | 0.985 / 0.992 | 0.913 | 15.70 (opendbc_0.3.1) | 2.69 | 0.1 | 1.16 / 1.21 / 1.42 / 1.54 / 1.77 / 1.85 / 1.55 |
| HONDA_ODYSSEY | LARGE | 5.14 | train | 5 | 2.1 | 76942 | 1.327 | 1.156 | 0.927 | 1.256 | 0.980 / 0.991 | 0.917 | 14.35 (opendbc_0.3.1) | 3.00 | 7.3 | 1.19 / 1.21 / 1.45 / 1.66 / 1.78 / 2.05 / 2.37 |
| HYUNDAI_TUCSON_4TH_GEN | MID | 4.53 | train | 10 | 16.0 | 576785 | 1.332 | 1.136 | 0.832 | 1.068 | 0.987 / 0.995 | 0.942 | 13.70 (opendbc_0.3.1) | 2.76 | 2.9 | 1.16 / 1.20 / 1.32 / 1.46 / 1.62 / 1.80 / 2.04 |
| HONDA_RIDGELINE | LARGE | 5.33 | train | 9 | 1.6 | 58204 | 1.337 | 1.177 | 0.796 | 1.191 | 0.979 / 0.992 | 0.887 | 15.59 (opendbc_0.3.1) | 3.18 | 1.6 | 1.21 / 1.22 / 1.39 / 1.59 / 1.67 / 1.98 / 2.26 |
| HYUNDAI_ELANTRA_2021 | SMALL | 4.31 | train | 10 | 5.2 | 185609 | 1.343 | 1.101 | 0.994 | 1.256 | 0.987 / 0.992 | 0.896 | 12.90 (opendbc_0.3.1) | 2.72 | 2.2 | 1.13 / 1.16 / 1.33 / 1.49 / 1.63 / 1.95 / 2.42 |
| KIA_NIRO_EV_2ND_GEN | SMALL | 4.26 | train | 9 | 4.0 | 144441 | 1.348 | 1.112 | 0.911 | 1.164 | 0.986 / 0.992 | 0.922 | 13.60 (opendbc_0.3.1) | 2.70 | 0.3 | 1.13 / 1.17 / 1.29 / 1.41 / 1.54 / 1.91 / 2.34 |
| TOYOTA_PRIUS_TSS2 | SMALL | 4.14 | train | 13 | 3.2 | 116967 | 1.396 | 1.158 | 0.981 | 1.302 | 0.980 / 0.988 | 0.913 | 13.40 (opendbc_0.3.1) | 2.70 | 1.2 | 1.15 / 1.21 / 1.31 / 1.43 / 1.69 / 2.07 / 2.51 |
| KIA_SPORTAGE_5TH_GEN | SMALL | 4.42 | train | 7 | 1.8 | 64322 | 1.459 | 1.093 | 1.046 | 1.346 | 0.988 / 0.992 | 0.933 | 13.60 (opendbc_0.3.1) | 2.76 | 0.8 | 1.11 / 1.16 / 1.34 / 1.52 / 1.67 / 1.77 / 2.10 |
| KIA_CARNIVAL_4TH_GEN | LARGE | 5.30 | held-out test | 9 | 1.8 | 63155 | 1.488 | 1.175 | 0.835 | 1.241 | 0.980 / 0.993 | 0.900 | 14.23 (opendbc_0.3.1) | 3.09 | 1.9 | 1.19 / 1.20 / 1.31 / 1.43 / 1.71 / 1.98 / 2.39 |
| HYUNDAI_KONA | SMALL | 4.02 | held-out test | 2 | 4.9 | 177036 | 1.524 | 1.141 | 0.729 | 1.233 | 0.988 / 0.996 | 0.875 | 13.42 (opendbc_0.3.1) | 2.60 | 0.2 | 1.15 / 1.20 / 1.27 / 1.36 / 1.55 / 1.86 / 2.39 |
| HYUNDAI_ELANTRA_HEV_2021 | SMALL | 4.31 | train | 4 | 9.4 | 338856 | 1.597 | 1.103 | 1.055 | 1.440 | 0.989 / 0.995 | 0.889 | 12.90 (opendbc_0.3.1) | 2.72 | 1.3 | 1.13 / 1.18 / 1.36 / 1.50 / 1.64 / 1.89 / 2.24 |
| TOYOTA_SIENNA | LARGE | 5.21 | held-out test | 3 | 1.2 | 43569 | 1.773 | 1.198 | 1.140 | 1.521 | 0.976 / 0.984 | 0.951 | 15.50 (opendbc_0.3.1) | 3.03 | 2.0 | 1.22 / 1.29 / 1.53 / 1.68 / 1.93 / 2.11 / 2.29 |
| TESLA_MODEL_X | LARGE | 5.06 | train | 2 | 2.5 | 89987 | 1.929 | 1.236 | 0.888 | 1.575 | 0.976 / 0.994 | 0.885 | 12.00 (opendbc_0.3.1) | 2.96 | 0.4 | 1.28 / 1.33 / 1.48 / 1.70 / 1.84 / 2.03 / 2.26 |
| TOYOTA_CHR_TSS2 | SMALL | 4.08 | validation | 3 | 2.1 | 74357 | 1.989 | 1.147 | 1.522 | 1.917 | 0.979 / 0.987 | 0.930 | 13.60 (opendbc_0.3.1) | 2.64 | 0.9 | 1.19 / 1.27 / 1.40 / 1.61 / 2.09 / 2.69 / 3.26 |
| RAM_HD_5TH_GEN | LARGE | 6.57 | train | 7 | 12.3 | 442908 | 2.038 | 1.046 | 1.884 | 1.850 | 0.912 / 0.913 | 0.990 | 15.61 (opendbc_0.3.1) | 3.79 | 0.4 | 1.09 / 1.20 / 1.51 / 1.91 / 2.18 / 2.60 / 3.18 |
| ACURA_RDX_3G | MID | 4.50 | train | 7 | 6.3 | 227001 | 2.421 | 1.227 | 1.350 | 2.016 | 0.968 / 0.987 | 0.918 | 11.95 (opendbc_0.3.1) | 2.75 | 0.6 | 1.28 / 1.37 / 1.65 / 1.78 / 1.98 / 2.50 / 2.93 |
| LEXUS_RX | MID | 4.55 | held-out test | 4 | 2.5 | 89974 | 2.505 | 1.052 | 1.737 | 2.279 | 0.990 / 0.992 | 0.918 | 16.00 (opendbc_0.3.1) | 2.79 | 2.8 | 1.06 / 1.14 / 1.42 / 1.63 / 2.40 / 2.85 / 2.57 |

## 8 comma2k19 validation

Median ratio of the derived yaw (corrected as stated) to the logged 2018 `carState.yawRate`, per speed bin; RAV4 188 segments,
Civic 202; `subset10` = 10 evenly spaced segments, `all` = every segment, `all_human` = the samples of all segments with the
2018 controller disengaged. `own_k` = the car's own lab k1 (the RAV4's 0.000837; the Civic platform of these logs is not fitted
in the lab). `model_2018_ak` = the (a, k) 2018 openpilot's vehicle model implies (RAV4 a 0.966, k 0.001144; Civic a 0.951,
k 0.000621).

| chunk / car | samples | yaw variant | median ratio derived / logged per bin 1-5 / 5-10 / 10-15 / 15-20 / 20-25 / 25-30 / 30-40 m/s |
|---|---|---|---|
| Chunk_1 (TOYOTA_RAV4, subset10) | 9778 | uncorrected | 0.99 / 1.03 / 1.17 / 1.24 / 1.45 / 1.72 / 2.07 |
| Chunk_1 (TOYOTA_RAV4, subset10) | 9778 | global_k | 0.97 / 0.98 / 1.02 / 1.04 / 1.08 / 1.12 / 1.15 |
| Chunk_1 (TOYOTA_RAV4, subset10) | 9778 | own_k | 0.97 / 0.98 / 1.01 / 1.03 / 1.06 / 1.09 / 1.13 |
| Chunk_1 (TOYOTA_RAV4, subset10) | 9778 | model_2018_ak | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| Chunk_1 (TOYOTA_RAV4, all) | 218479 | uncorrected | 0.98 / 1.03 / 1.12 / 1.27 / 1.54 / 1.85 / 2.06 |
| Chunk_1 (TOYOTA_RAV4, all) | 218479 | global_k | 0.97 / 0.99 / 1.01 / 1.04 / 1.09 / 1.13 / 1.15 |
| Chunk_1 (TOYOTA_RAV4, all) | 218479 | own_k | 0.97 / 0.98 / 1.00 / 1.03 / 1.07 / 1.11 / 1.13 |
| Chunk_1 (TOYOTA_RAV4, all) | 218479 | model_2018_ak | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| Chunk_1 (TOYOTA_RAV4, all_human) | 102072 | uncorrected | 0.98 / 1.03 / 1.12 / 1.27 / 1.55 / 1.84 / 2.07 |
| Chunk_1 (TOYOTA_RAV4, all_human) | 102072 | global_k | 0.97 / 0.99 / 1.01 / 1.04 / 1.09 / 1.13 / 1.15 |
| Chunk_1 (TOYOTA_RAV4, all_human) | 102072 | own_k | 0.97 / 0.98 / 1.00 / 1.03 / 1.07 / 1.11 / 1.13 |
| Chunk_1 (TOYOTA_RAV4, all_human) | 102072 | model_2018_ak | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| Chunk_3 (HONDA_CIVIC, subset10) | 9520 | uncorrected | 0.86 / 0.90 / 1.09 / 1.26 / 1.43 / 1.79 / 2.05 |
| Chunk_3 (HONDA_CIVIC, subset10) | 9520 | global_k | 0.86 / 0.87 / 0.94 / 0.99 / 1.03 / 1.10 / 1.13 |
| Chunk_3 (HONDA_CIVIC, subset10) | 9520 | own_k | platform not fitted in the lab |
| Chunk_3 (HONDA_CIVIC, subset10) | 9520 | model_2018_ak | 0.90 / 0.92 / 1.02 / 1.09 / 1.15 / 1.26 / 1.32 |
| Chunk_3 (HONDA_CIVIC, all) | 155922 | uncorrected | 0.86 / 0.91 / 1.03 / 1.15 / 1.52 / 1.78 / 2.01 |
| Chunk_3 (HONDA_CIVIC, all) | 155922 | global_k | 0.85 / 0.87 / 0.92 / 0.96 / 1.05 / 1.09 / 1.13 |
| Chunk_3 (HONDA_CIVIC, all) | 155922 | own_k | platform not fitted in the lab |
| Chunk_3 (HONDA_CIVIC, all) | 155922 | model_2018_ak | 0.90 / 0.92 / 0.99 / 1.04 / 1.18 / 1.25 / 1.31 |
| Chunk_3 (HONDA_CIVIC, all_human) | 67180 | uncorrected | 0.86 / 0.91 / 1.02 / 1.15 / 1.43 / 1.76 / 2.23 |
| Chunk_3 (HONDA_CIVIC, all_human) | 67180 | global_k | 0.85 / 0.87 / 0.91 / 0.96 / 1.03 / 1.09 / 1.16 |
| Chunk_3 (HONDA_CIVIC, all_human) | 67180 | own_k | platform not fitted in the lab |
| Chunk_3 (HONDA_CIVIC, all_human) | 67180 | model_2018_ak | 0.90 / 0.92 / 0.98 / 1.04 / 1.15 / 1.25 / 1.36 |

Reading it. With the global k the RAV4 ratios go from 0.98 - 2.06 (uncorrected, all segments) to 0.97 - 1.15 across the speed bins,
and the Civic's from 0.86 - 2.01 to 0.85 - 1.13; the bias that grew to a factor of two is reduced to a residual of about +15 % at
30-40 m/s (RAV4 +15.3 %, Civic +12.8 % on all segments; +15.4 % and +13.3 % on the ten-segment subsets) and below +5 % up to 20 m/s. The Civic's 0.86
at 1-5 m/s is unchanged by any k: it is the geometry error of section 5 (the 2-par fit on the Civic logs gives a = 0.852, i.e. an
implied steerRatio of about 13.1 instead of the constructed 15.38), not understeer. The two cars' own k fitted on comma2k19
itself (1-par k1 0.00106 RAV4, 0.00101 Civic; 2-par k 0.00113 / 0.00138) is larger than the global value and would fit them
slightly better - the lab's RAV4 k1 of 0.000837 already brings the top bin to 1.13 - but it is not used, because it would be
fitted on the external set itself. The `model_2018_ak` rows reproduce the RAV4 to 1.00 in every bin, which is the subject of the
first caveat below.

## 9 Caveats

* **The comma2k19 "measured" yaw is a model output.** The script's `yaw_is_model_test` reproduces the logged yawRate from the
  same steering angle with a v delta / (L (1 + k v^2)) to a median relative error of 3e-8 for both cars: the 2018 openpilot
  `carState.yawRate` is its single-track vehicle model, not an inertial sensor. The validation therefore checks that the lab-fitted
  global k agrees with the understeer that 2018 openpilot's own model assigned to these two cars (RAV4 k 0.00114; for the Civic the
  logged values imply a 0.845 / k 0.00140, not the a 0.951 / k 0.00062 reconstructed from the 2018 interface, so its
  `model_2018_ak` rows do not reach 1.00). The lab fit itself rests on the IMU / pose yaw.
* **Pose-yaw noise.** The lab truth is the released pose / IMU yaw (`yaw_valid` rows only). Noise in it is handled by the robust
  pass (3 scaled MADs; 86.9 - 99.0 % of the samples kept per platform) and by the |yaw_meas| >= 0.01 rad/s floor of the bin
  ratios; R^2 of the final fits is 0.91 - 1.00 (one platform, the RAM HD, at 0.91 / 0.91).
* **steerRatio and wheelbase errors.** They move a and k1 but not the 2-par k; the global value is built from k1, so platforms with a
  wrong geometry pull it in the direction of their a. The hours-weighted median limits that pull; the 2-par hours-weighted median
  (0.000752) differs from k_global by 5.2 %.
* **Angle offset.** The lab fit removes a per-drive constant offset estimated on straight driving (median |offset| 0.1 - 7.3 deg
  per platform); the decoder does not remove it on the external sets. The `raw` variant, which keeps the logged angle, gives
  k_global 0.000822 (+3.7 %), so the offset handling is not what sets the value.
* **Automation-engaged samples included.** The primary fit uses all samples, lateral controller on or off (`op_lat_active`,
  controller-on fraction per platform in the csv), because the external features are decoded from all samples; the human-only
  and controller-on subsets give 0.000815 and 0.000787. The comma2k19 `all_human` rows use `controlsState.enabled` of the 2018 logs.
* **Coverage.** 225 of the 361 lab platforms have usable samples and 78 are eligible (2 vehicles, 1 h); the global k is a median over
  62 platforms weighted by their hours, so the heavy platforms (Rivian R1 48.9 h + 25.7 h, Honda Civic Bosch 30.0 h, F-150 17.1 h,
  Model 3 16.8 h, Tucson 16.0 h, Ioniq 6 15.8 h) count most. The 2 vehicles / 1 h rule excludes the single-vehicle platforms.
* The correction uses the steady-state single-track factor; transients and the tan versus small-angle form are left as in the
  default decoder path (`common.yaw_from_steering`).

## 10 Reproduce

    source env.sh
    nice -n 10 $CPY experiments/transfer/understeer_fit.py --workers 6 --json experiments/transfer/understeer_fit.json

Scratch under `<data-root>/tmp/understeer/` (`platform_table.csv`, `platform_table.md`, `c2k_table.md`, `fit.json`, run log);
the `--json` copy is the same numbers-only JSON (platform names, no device / route identifiers, no machine paths). Options:
`--angle-variant raw`, `--c2k-npz-dir` (default `<data-root>/tmp/comma2k19/verify/logged`, the verifier's dumps), `--limit N`
for a debug run. The decoder option that consumes the result: `ccs_decode.py --yaw derived-understeer --understeer-k 0.00079289`
(CCS_PIPELINE.md, "Understeer-corrected yaw").
