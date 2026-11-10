# Diagnostic: derived acceleration and resampled speed on our own held-out test windows

Generated 9 Nov 2026 21:29 by experiments/transfer/diag_derived_accel.py. 8000 random test_full windows per task (seed 0) of the split_platform test split (held-out car models), scored by the seed ensembles of the reduced-input variants after replacing the input channels the way the external adapters have to build them. `recall` in class order; `drive` = mean probability per drive -> arg-max.

## size (SMALL / MID / LARGE): 8000 windows, 1058 drives

| variant | input treatment | window acc % | window F1 % | window recall % | window predicted share % | drive acc % | drive F1 % | drive recall % |
|---|---|---|---|---|---|---|---|---|
| V3 speed-only | CAN aEgo (as trained) | 48.0 | 42.5 | SMALL 58.7 / MID 38.7 / LARGE 30.6 | SMALL 50.2 / MID 31.0 / LARGE 18.8 | 52.2 | 45.0 | SMALL 70.3 / MID 35.1 / LARGE 28.7 |
| V3 speed-only | a := smooth_derivative(v), 0.5 s (ULTra-AV / OpenACC / microSIM / UGA / A2D2 / Tesla) | 45.0 | 33.7 | SMALL 63.3 / MID 35.9 / LARGE 6.8 | SMALL 61.5 / MID 33.7 / LARGE 4.8 | 43.5 | 32.9 | SMALL 58.6 / MID 47.5 / LARGE 3.4 |
| V3 speed-only | a := smooth_derivative(v), 1.0 s | 48.6 | 34.4 | SMALL 73.8 / MID 27.5 / LARGE 6.9 | SMALL 71.5 / MID 23.2 / LARGE 5.2 | 46.9 | 32.8 | SMALL 72.8 / MID 31.7 / LARGE 3.0 |
| V3 speed-only | v smoothed 0.5 s + derived a (GNSS-spline-like) | 47.6 | 32.5 | SMALL 73.4 / MID 26.7 / LARGE 4.2 | SMALL 71.9 / MID 23.7 / LARGE 4.4 | 45.5 | 30.1 | SMALL 72.8 / MID 27.9 / LARGE 0.8 |
| V3 speed-only | v held at 1 Hz + derived a, 1.0 s (VED-like) | 49.2 | 31.1 | SMALL 79.7 / MID 20.6 / LARGE 2.3 | SMALL 78.8 / MID 17.3 / LARGE 3.9 | 48.0 | 29.7 | SMALL 82.0 / MID 18.1 / LARGE 1.7 |
| V2 longitudinal-only | CAN aEgo (as trained) | 41.6 | 34.8 | SMALL 56.0 / MID 18.1 / LARGE 31.4 | SMALL 55.5 / MID 20.9 / LARGE 23.6 | 44.1 | 34.9 | SMALL 66.2 / MID 13.6 / LARGE 26.6 |
| V2 longitudinal-only | a := smooth_derivative(v), 0.5 s (ULTra-AV / OpenACC / microSIM / UGA / A2D2 / Tesla) | 42.2 | 33.8 | SMALL 60.0 / MID 17.1 / LARGE 25.1 | SMALL 59.9 / MID 20.0 / LARGE 20.2 | 43.6 | 31.2 | SMALL 70.9 / MID 11.7 / LARGE 15.2 |
| V2 longitudinal-only | a := smooth_derivative(v), 1.0 s | 41.8 | 33.3 | SMALL 59.7 / MID 15.1 / LARGE 26.4 | SMALL 60.1 / MID 18.4 / LARGE 21.6 | 43.4 | 32.1 | SMALL 68.9 / MID 10.9 / LARGE 19.8 |
| V2 longitudinal-only | v smoothed 0.5 s + derived a (GNSS-spline-like) | 42.2 | 33.5 | SMALL 60.5 / MID 16.1 / LARGE 24.8 | SMALL 60.7 / MID 19.0 / LARGE 20.2 | 44.5 | 32.1 | SMALL 72.1 / MID 12.1 / LARGE 16.0 |
| V2 longitudinal-only | v held at 1 Hz + derived a, 1.0 s (VED-like) | 42.0 | 32.9 | SMALL 60.6 / MID 12.3 / LARGE 28.4 | SMALL 60.6 / MID 15.5 / LARGE 23.9 | 43.6 | 30.9 | SMALL 70.7 / MID 7.2 / LARGE 20.7 |

## bev_ice (BEV / ICE): 8000 windows, 802 drives

| variant | input treatment | window acc % | window F1 % | window recall % | window predicted share % | drive acc % | drive F1 % | drive recall % |
|---|---|---|---|---|---|---|---|---|
| V3 speed-only | CAN aEgo (as trained) | 77.1 | 74.7 | BEV 68.7 / ICE 81.4 | BEV 35.7 / ICE 64.3 | 86.8 | 85.8 | BEV 76.5 / ICE 93.4 |
| V3 speed-only | a := smooth_derivative(v), 0.5 s (ULTra-AV / OpenACC / microSIM / UGA / A2D2 / Tesla) | 72.2 | 64.6 | BEV 38.2 / ICE 89.7 | BEV 19.8 / ICE 80.2 | 74.7 | 70.1 | BEV 45.1 / ICE 93.8 |
| V3 speed-only | a := smooth_derivative(v), 1.0 s | 69.6 | 64.1 | BEV 44.8 / ICE 82.3 | BEV 26.9 / ICE 73.1 | 72.6 | 70.0 | BEV 54.9 / ICE 84.0 |
| V3 speed-only | v smoothed 0.5 s + derived a (GNSS-spline-like) | 71.1 | 64.5 | BEV 41.1 / ICE 86.6 | BEV 22.8 / ICE 77.2 | 72.9 | 69.4 | BEV 49.5 / ICE 88.1 |
| V3 speed-only | v held at 1 Hz + derived a, 1.0 s (VED-like) | 64.5 | 59.8 | BEV 44.5 / ICE 74.8 | BEV 31.8 / ICE 68.2 | 65.0 | 63.4 | BEV 56.5 / ICE 70.4 |
| V2 longitudinal-only | CAN aEgo (as trained) | 76.8 | 74.3 | BEV 66.7 / ICE 82.0 | BEV 34.5 / ICE 65.5 | 87.4 | 86.4 | BEV 75.9 / ICE 94.9 |
| V2 longitudinal-only | a := smooth_derivative(v), 0.5 s (ULTra-AV / OpenACC / microSIM / UGA / A2D2 / Tesla) | 72.8 | 70.9 | BEV 69.1 / ICE 74.8 | BEV 40.2 / ICE 59.8 | 86.3 | 85.7 | BEV 83.8 / ICE 87.9 |
| V2 longitudinal-only | a := smooth_derivative(v), 1.0 s | 60.9 | 60.4 | BEV 73.2 / ICE 54.5 | BEV 54.9 / ICE 45.1 | 63.1 | 62.9 | BEV 89.5 / ICE 46.0 |
| V2 longitudinal-only | v smoothed 0.5 s + derived a (GNSS-spline-like) | 69.4 | 68.0 | BEV 71.4 / ICE 68.4 | BEV 45.2 / ICE 54.8 | 80.5 | 80.2 | BEV 86.7 / ICE 76.6 |
| V2 longitudinal-only | v held at 1 Hz + derived a, 1.0 s (VED-like) | 54.2 | 54.2 | BEV 74.3 / ICE 43.9 | BEV 62.3 / ICE 37.7 | 52.7 | 50.9 | BEV 91.7 / ICE 27.5 |

