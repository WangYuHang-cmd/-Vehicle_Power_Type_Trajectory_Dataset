# Feature launch_accel_1s_mps2 (lens: longitudinal_powertrain) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Acceleration 0.8-1.2 s after a human tip-in from (near) standstill (v < 0.5 m/s, brake released, gas held >= 0.8 s, no automation). An electric motor gives full torque from zero speed, so the car is already at its launch acceleration after one second; an ICE automatic first needs engine rpm to rise, converter slip to build torque and, for turbo engines, boost to develop. Heavier vehicles accelerate less for the same launch torque.
# Hypothesis: BEV > ICE; LARGE < MID < SMALL
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled
# Training-sample statistics: finite 0.15, AUROC BEV-vs-ICE 0.523, Spearman(footprint) -0.004
# Audit: keep_with_caveat - Physically motivated (motor torque from zero rpm) but at 1 s the value is set by how hard the driver presses: weak (AUROC 0.52) and no size trend within ICE (rho -0.05) or BEV (0.01). Keep only as a driver-habit-laden launch cue; the classifier can down-weight it.
import numpy as np

def feat_launch_accel_1s_mps2(w):
    """Median acceleration 0.8-1.2 s after a human tip-in from (near) standstill (v < 0.5 m/s, brake released, gas held >= 0.8 s)."""
    v = np.asarray(w["v_mps"], dtype=np.float64)
    a = np.asarray(w["a_mps2"], dtype=np.float64).copy()
    a[np.abs(a) > 8.0] = np.nan
    gi = np.nan_to_num(np.asarray(w["gas_pressed"], dtype=np.float64), nan=0.0) > 0.5
    bi = np.nan_to_num(np.asarray(w["brake_pressed"], dtype=np.float64), nan=1.0) > 0.5
    di = np.nan_to_num(np.asarray(w["das_enabled"], dtype=np.float64), nan=1.0) > 0.5
    n = a.shape[0]
    vals = []
    for i in range(5, n - 15):
        if not (gi[i] and not gi[i - 1]):
            continue
        if not np.isfinite(v[i]) or v[i] < -0.1 or v[i] >= 0.5:
            continue
        if not gi[i:i + 8].all():
            continue
        if bi[i - 2:i + 13].any() or di[i - 5:i + 15].any():
            continue
        seg = a[i + 8:i + 13]
        if np.isfinite(seg).sum() < 3:
            continue
        vals.append(np.nanmean(seg))
    if len(vals) < 1:
        return float("nan")
    return float(np.median(vals))
