# Feature torque_per_deg_lowspeed_human (lens: lateral_inertia_geometry) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Steering effort per unit hand-wheel angle while the human manoeuvres at 1-6 m/s (das off, steer_pressed on): driver torque per degree equals (torque per unit curvature, set by front-axle scrub/aligning moment and speed-dependent EPS assist) times (curvature per degree, 1/(L i_s)). The torque is normalised by the drive's own 95th-percentile effort; because EPS assist is tuned to a similar hand-wheel feel per unit curvature across cars, the per-degree effort inherits the steering geometry from an independent sensor (driver torque instead of the IMU).
# Hypothesis: SMALL > MID > LARGE; BEV vs ICE uncertain. Training sample: S/M/L 3.26e-3 / 2.98e-3 / 2.37e-3 per deg (Spearman -0.20); BEV 2.64e-3 < ICE 3.12e-3 (AUROC 0.43).
# Channels: v_mps, steer_torque_norm, steer_angle_deg, das_enabled, steer_pressed
# Training-sample statistics: finite 0.16, AUROC BEV-vs-ICE 0.434, Spearman(footprint) -0.196
# Audit: keep_with_caveat - Measures manoeuvring effort per hand-wheel degree (EPS assist map times 1/(i_s L)); renormalising torque within the window leaves it unchanged (AUROC 0.447 vs 0.434, size rho -0.18 vs -0.20), so the drive-level normaliser is not its source. Caveats: EPS assist tuning is a design choice rather than dynamics, the assumed 'similar feel per curvature across cars' is only approximately true, and the driver's manoeuvre mix enters.
import numpy as np

def feat_torque_per_deg_lowspeed_human(w):
    """normalised driver torque per hand-wheel degree during human steering at 1-6 m/s (manoeuvring effort per degree)."""
    v = w["v_mps"]; tq = w["steer_torque_norm"]; d = w["steer_angle_deg"]
    m = (w["das_enabled"] < 0.5) & (w["steer_pressed"] > 0.5) & (v >= 1.0) & (v <= 6.0) & np.isfinite(tq) & np.isfinite(d)
    if m.sum() < 30:
        return float("nan")
    x = d[m].astype(float); y = tq[m].astype(float)
    if np.std(x) < 15.0:
        return float("nan")
    xc = x - x.mean(); yc = y - y.mean(); den = float(np.sum(xc * xc))
    if den < 1e-9:
        return float("nan")
    s = float(np.sum(xc * yc) / den)
    if not np.isfinite(s) or s <= 0.0 or s > 0.1:
        return float("nan")
    return s
