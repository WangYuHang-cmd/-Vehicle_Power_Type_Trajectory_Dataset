# Feature drive_specific_power_p95 (lens: energy_speed_profile) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Usable power-to-mass: p95 of specific tractive power v*a (W/kg) over human-driven accelerating samples above 5 m/s with the brake released. Beyond the base speed a drivetrain is power-limited, a = P/(m*v) minus losses, so v*a is the vehicle's specific power actually delivered. BEVs (especially the long-range models in openpilot fleets) have higher continuous specific power and deliver it without downshift delay, so drivers reach higher v*a in ordinary driving; heavy three-row ICE vehicles have the lowest specific power. Not conditioned on the accelerator flag for the one-pedal reason given above.
# Hypothesis: BEV > ICE; LARGE < SMALL/MID (lower specific power of large family vehicles, though powerful large BEVs blur this).
# Channels: v_mps, a_mps2, das_enabled, brake_pressed
# Training-sample statistics: finite 0.47, AUROC BEV-vs-ICE 0.560, Spearman(footprint) 0.013
# Audit: keep_with_caveat - Bounded by vehicle specific power but far from it: medians 14.5 (BEV) vs 12.1 (ICE) W/kg against capabilities of 50-150 W/kg, so the p95 of ordinary driving measures driver demand and traffic (Spearman +0.24 with mean speed, +0.31 within ICE; AUROC 0.56, rho_fp 0.01). Keep only as a weak behavioural covariate; a capability cue needs the speed dependence of the acceleration envelope (see gaps).
import numpy as np

def feat_drive_specific_power_p95(w):
    v = w["v_mps"]; a = w["a_mps2"]; das = w["das_enabled"]; brk = w["brake_pressed"]
    m = (np.isfinite(v) & np.isfinite(a) & (v > 5.0) & (a > 0.0)
         & (das < 0.5) & (brk < 0.5))
    if np.count_nonzero(m) < 30:
        return float("nan")
    out = float(np.nanpercentile(v[m] * a[m], 95))
    return out if np.isfinite(out) else float("nan")
