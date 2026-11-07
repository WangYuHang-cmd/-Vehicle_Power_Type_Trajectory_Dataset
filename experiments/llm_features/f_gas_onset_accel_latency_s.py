# Feature gas_onset_accel_latency_s (lens: longitudinal_powertrain) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Delay between the accelerator flag rising and the body acceleration rising 0.25 m/s^2 above its pre-tip-in baseline, for clean human tip-ins (pedal-free >= 1 s before, gas held >= 1 s, no brake, no automation, and a real response of >= 0.6 m/s^2 within 2.5 s). An electric motor develops torque within tens of milliseconds; a combustion drivetrain adds intake-manifold filling, torque-converter slip, a downshift decision and turbo spool (0.3-1 s). A heavier vehicle needs more torque per unit of acceleration, so it crosses the threshold later.
# Hypothesis: BEV < ICE; LARGE > MID > SMALL
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled
# Training-sample statistics: finite 0.29, AUROC BEV-vs-ICE 0.469, Spearman(footprint) 0.148
# Audit: keep_with_caveat - Sound (torque build-up delay after a tip-in), moving-tip-in design with a baseline-relative threshold avoids the launch dead zone. Caveats: gas_pressed is a port-defined threshold flag and the driver's pedal ramp rate adds to the delay; the within-BEV size relation (rho 0.31) comes from one LARGE-BEV group at 0.6 s vs 0.3 s, i.e. pedal-map tuning of a platform; 0.1-s quantised; BEV/ICE separation weak (0.47).
import numpy as np

def feat_gas_onset_accel_latency_s(w):
    """Median delay (s) from an accelerator tip-in to a +0.25 m/s^2 rise of acceleration above the pre-tip-in baseline (>= 1 event)."""
    v = np.asarray(w["v_mps"], dtype=np.float64)
    a = np.asarray(w["a_mps2"], dtype=np.float64).copy()
    a[np.abs(a) > 8.0] = np.nan
    gi = np.nan_to_num(np.asarray(w["gas_pressed"], dtype=np.float64), nan=0.0) > 0.5
    bi = np.nan_to_num(np.asarray(w["brake_pressed"], dtype=np.float64), nan=1.0) > 0.5
    di = np.nan_to_num(np.asarray(w["das_enabled"], dtype=np.float64), nan=1.0) > 0.5
    n = a.shape[0]
    lat = []
    for i in range(10, n - 25):
        if not (gi[i] and not gi[i - 1]):
            continue
        if gi[i - 10:i].any() or not gi[i:i + 10].all():
            continue
        if bi[i - 5:i + 15].any() or di[i - 10:i + 25].any():
            continue
        if np.nanmin(v[i - 5:i + 25]) < -0.1:
            continue
        pre = a[i - 5:i]
        if np.isnan(pre).sum() > 2:
            continue
        seg = a[i:i + 25] - np.nanmedian(pre)
        if np.isnan(seg).sum() > 5 or np.nanmax(seg) < 0.6:
            continue
        hit = np.where(seg >= 0.25)[0]
        if hit.size == 0:
            continue
        lat.append(0.1 * hit[0])
    if len(lat) < 1:
        return float("nan")
    return float(np.median(lat))
