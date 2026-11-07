# Feature curv_gain_midspeed (lens: lateral_inertia_geometry) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Curvature-per-degree transfer gain in the 6-18 m/s band, where the single-track model gives kappa/delta = 1/(i_s (L + K_us v^2/g)): the understeer term (0.1-1.2 m) is a fraction of the wheelbase, so the feature mixes geometry (L, i_s) with the understeer gradient. Battery-electric cars carry the pack in the floor between the axles (near 50/50 axle load, lower understeer gradient than nose-heavy front-engine cars) and several BEV platforms use quick racks, so their gain should be higher at equal footprint. Curvature is advanced 0.2 s for the yaw response lag; OLS with intercept absorbs sensor offsets.
# Hypothesis: SMALL > MID > LARGE; BEV > ICE. Training sample (final code): S/M/L medians 3.76e-4 / 3.40e-4 / 3.11e-4 1/(m deg) (Spearman -0.48); BEV 3.45e-4 vs ICE 3.31e-4 (AUROC 0.56).
# Channels: v_mps, steer_angle_deg, curv_1pm, standstill
# Training-sample statistics: finite 0.42, AUROC BEV-vs-ICE 0.565, Spearman(footprint) -0.480
# Audit: keep_with_caveat - The one solid size feature: rho_fp -0.48, and between-model spread (MAD of pseudo-model medians 3.2e-5) exceeds within-model spread (1.7e-5), so it identifies the vehicle. Caveats: (i) as written it pools human- and automation-steered samples; the DAS-steered subset alone has BEV/ICE AUROC 0.71 versus 0.51 for human-steered samples with the same rho_fp (-0.41 / -0.48), so the small powertrain signal (0.565) comes from the controller-steered regime (EPS torque twisting the torsion bar upstream of the angle sensor, make-specific minimum steer speeds), not geometry - condition on das_enabled < 0.5 (or steer_pressed) to keep it a vehicle property; (ii) the 6-18 m/s band mixes 1/(i_s L) with the understeer term (Spearman -0.25 with band speed); a 2-6 m/s Ackermann variant is cleaner (rho_fp -0.67, BEV/ICE 0.49).
import numpy as np

def feat_curv_gain_midspeed(w):
    """kappa per hand-wheel degree at 6-18 m/s ~ 1/(i_s*(L + K_us*v^2/g)); geometry plus a first understeer contribution.
    FIXED (verifier): band 8-15 -> 6-18 m/s (understeer term 0.1-1.2 m, same regime), min samples 60 -> 30,
    hand-wheel std gate 4 -> 2 deg, |delta| gate 3 -> 1 deg; coverage 0.20 -> 0.42, same class ordering."""
    v = w["v_mps"]; d = w["steer_angle_deg"]; k0 = w["curv_1pm"]; ss = w["standstill"]
    k = np.full(k0.shape, np.nan); k[:-2] = k0[2:]
    m = (v >= 6.0) & (v <= 18.0) & (ss < 0.5) & np.isfinite(k) & np.isfinite(d) & (np.abs(d) >= 1.0)
    if m.sum() < 30:
        return float("nan")
    x = d[m].astype(float); y = k[m].astype(float)
    if np.std(x) < 2.0:
        return float("nan")
    xc = x - x.mean(); yc = y - y.mean(); den = float(np.sum(xc * xc))
    if den < 1e-9:
        return float("nan")
    s = float(np.sum(xc * yc) / den)
    if not np.isfinite(s) or s <= 1e-5 or s > 2e-3:
        return float("nan")
    return s
