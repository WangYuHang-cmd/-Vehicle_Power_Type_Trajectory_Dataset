# Feature steer_gain_angle_nonlinearity (lens: lateral_inertia_geometry) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Steering geometry beyond the linear range: kappa = tan(delta_hw / i_s) / L, so curvature per hand-wheel degree rises with angle, and the rise is larger for a quick rack and a short wheelbase; variable-ratio racks (common on compact combustion cars) add to it. The feature is the ratio of the low-speed (2-9 m/s) median curvature-per-degree at |delta| >= 100 deg to that at 10-50 deg; the ratio cancels the static gain.
# Hypothesis: SMALL > MID > LARGE; ICE > BEV. Training sample: S/M/L 1.071 / 1.052 / 1.029 (Spearman -0.23); BEV 1.036 < ICE 1.073 (AUROC 0.38 for BEV > ICE).
# Channels: v_mps, steer_angle_deg, curv_1pm, standstill
# Training-sample statistics: finite 0.14, AUROC BEV-vs-ICE 0.381, Spearman(footprint) -0.226
# Audit: keep_with_caveat - Vehicle-level steering nonlinearity, but not the mechanism stated: the 3-7% ratios are an order of magnitude larger than the tan() effect (<1% at 120 deg hand-wheel), so the measurement reflects variable-ratio racks and the lower speed (less understeer) at large angles. Both are vehicle properties, so keep, with the caveat that the manoeuvre mix (parking vs turns) also moves the ratio; ICE > BEV (AUROC 0.38) and SMALL > LARGE (rho -0.23) are plausible.
import numpy as np

def feat_steer_gain_angle_nonlinearity(w):
    """low-speed curvature-per-degree at |delta| >= 100 deg divided by that at 10-50 deg: Ackermann tan() growth and variable-ratio racks."""
    v = w["v_mps"]; d = w["steer_angle_deg"]; k0 = w["curv_1pm"]; ss = w["standstill"]
    k = np.full(k0.shape, np.nan); k[:-2] = k0[2:]
    base = (v >= 2.0) & (v <= 9.0) & (ss < 0.5) & np.isfinite(k) & np.isfinite(d)
    small = base & (np.abs(d) >= 10.0) & (np.abs(d) < 50.0)
    large = base & (np.abs(d) >= 100.0)
    if small.sum() < 20 or large.sum() < 20:
        return float("nan")
    gs = float(np.nanmedian(k[small] / d[small])); gl = float(np.nanmedian(k[large] / d[large]))
    if not (np.isfinite(gs) and np.isfinite(gl)) or gs <= 1e-5:
        return float("nan")
    ratio = gl / gs
    if ratio < 0.3 or ratio > 3.0:
        return float("nan")
    return float(ratio)
