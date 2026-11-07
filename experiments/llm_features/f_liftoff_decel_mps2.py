# Feature liftoff_decel_mps2 (lens: longitudinal_powertrain) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Acceleration 0.6-1.2 s after the accelerator is released at >= 5 m/s, with no pedal and no automation for 1.3 s afterwards. This is the drivetrain's lift-off torque: regenerative braking in a BEV (-1 to -2 m/s^2 in one-pedal settings, present within 0.3 s) versus engine braking plus drag in an ICE (-0.2 to -0.5 m/s^2, building slowly because fuel cut-off and torque-converter behaviour are delayed).
# Hypothesis: BEV < ICE (more negative)
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled
# Training-sample statistics: finite 0.22, AUROC BEV-vs-ICE 0.149, Spearman(footprint) -0.021
# Audit: keep_with_caveat - Strongest regen feature where defined (BEV -1.19 vs ICE -0.26 m/s^2, AUROC 0.15; PHEV -0.48 in between). Caveats: regen level is a driver setting; grade is not differenced out; redundant with liftoff_decel_moving (0.86) and pre_brake_decel (0.62) but a distinct event-based estimator.
import numpy as np

def feat_liftoff_decel_mps2(w):
    """Median acceleration 0.6-1.2 s after accelerator release (held >= 0.5 s before, no pedal/automation 1.3 s after, v >= 5 m/s), >= 1 event."""
    v = np.asarray(w["v_mps"], dtype=np.float64)
    a = np.asarray(w["a_mps2"], dtype=np.float64).copy()
    a[np.abs(a) > 8.0] = np.nan
    gi = np.nan_to_num(np.asarray(w["gas_pressed"], dtype=np.float64), nan=0.0) > 0.5
    bi = np.nan_to_num(np.asarray(w["brake_pressed"], dtype=np.float64), nan=1.0) > 0.5
    di = np.nan_to_num(np.asarray(w["das_enabled"], dtype=np.float64), nan=1.0) > 0.5
    n = a.shape[0]
    vals = []
    for i in range(10, n - 13):
        if not (gi[i - 1] and not gi[i]):
            continue
        if not gi[i - 5:i].all() or gi[i:i + 13].any():
            continue
        if bi[i - 3:i + 13].any() or di[i - 10:i + 13].any():
            continue
        if not np.isfinite(v[i]) or v[i] < 5.0:
            continue
        seg = a[i + 6:i + 13]
        if np.isfinite(seg).sum() < 4:
            continue
        vals.append(np.nanmean(seg))
    if len(vals) < 1:
        return float("nan")
    return float(np.median(vals))
