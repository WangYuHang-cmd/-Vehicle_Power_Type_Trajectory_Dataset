# Feature curv_gain_lowspeed (lens: lateral_inertia_geometry) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Kinematic (Ackermann) steering: at low speed kappa = delta_hw / (i_s * L), so the OLS slope of curvature on hand-wheel angle (intercept absorbs sensor offsets) measures 1/(wheelbase x steering ratio). Larger vehicles have a longer wheelbase and usually a slower rack, so the gain falls with footprint. Only 2.5-9 m/s is used so the understeer term stays negligible; curvature is advanced 0.2 s for the yaw response lag. Who steers does not matter because it is a vehicle transfer gain.
# Hypothesis: SMALL > MID > LARGE; BEV ~ ICE. Training sample: S/M/L medians 4.29e-4 / 3.75e-4 / 3.25e-4 1/(m deg), Spearman -0.69 vs footprint; BEV 3.70e-4 vs ICE 3.72e-4 (AUROC 0.48).
# Channels: v_mps, steer_angle_deg, curv_1pm, standstill
# Training-sample statistics: finite 0.26, AUROC BEV-vs-ICE 0.477, Spearman(footprint) -0.691
# Audit: keep - Kinematic steering gain 1/(i_s L) with the 0.2-s yaw-lag shift and OLS with intercept; vehicle transfer gain, so it needs no human conditioning. Strongest size cue of all candidates (footprint rho -0.69, within-ICE -0.81) and powertrain-neutral as hypothesised (AUROC 0.48). Redundant with ackermann_gain_2_6 (0.90) and the inverse of steer_geom_gain (-0.78) but a distinct estimator, not a duplicate.
import numpy as np

def feat_curv_gain_lowspeed(w):
    """kappa per hand-wheel degree at 2.5-9 m/s ~ 1/(L*i_s); yaw/curvature shifted 0.2 s to absorb the yaw response lag."""
    v = w["v_mps"]; d = w["steer_angle_deg"]; k0 = w["curv_1pm"]; ss = w["standstill"]
    k = np.full(k0.shape, np.nan); k[:-2] = k0[2:]
    m = (v >= 2.5) & (v <= 9.0) & (ss < 0.5) & np.isfinite(k) & np.isfinite(d) & (np.abs(d) >= 8.0)
    if m.sum() < 30:
        return float("nan")
    x = d[m].astype(float); y = k[m].astype(float)
    if np.std(x) < 10.0:
        return float("nan")
    xc = x - x.mean(); yc = y - y.mean(); den = float(np.sum(xc * xc))
    if den < 1e-9:
        return float("nan")
    s = float(np.sum(xc * yc) / den)
    if not np.isfinite(s) or s <= 1e-5 or s > 2e-3:
        return float("nan")
    return s
