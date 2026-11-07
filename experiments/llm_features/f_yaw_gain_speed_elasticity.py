# Feature yaw_gain_speed_elasticity (lens: lateral_inertia_geometry) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Understeer proxy from the speed dependence of the yaw-rate gain: r/delta = v / (i_s (L + K_us v^2/g)), so d ln(r/delta)/d ln v = 1 - 2 K_us v^2 / (g L + K_us v^2). A neutral-steering car gives +1; the elasticity falls towards -1 as understeer grows. The yaw-rate gain is estimated by OLS with intercept in 4 m/s bands between 8 and 37 m/s and the log-gains regressed on log mean speed, which cancels the unknown steering ratio. Floor-battery BEVs with an even axle load split are closer to neutral than nose-heavy ICE cars; SUVs and trucks are tuned with more understeer.
# Hypothesis: BEV > ICE (closer to +1); LARGE < SMALL expected but not visible. Training sample: BEV 0.79 vs ICE 0.67 (AUROC 0.57); S/M/L 0.68 / 0.75 / 0.77 (flat, Spearman +0.06).
# Channels: v_mps, steer_angle_deg, yaw_rate_radps
# Training-sample statistics: finite 0.12, AUROC BEV-vs-ICE 0.572, Spearman(footprint) 0.059
# Audit: keep - Understeer proxy from the speed dependence of the yaw-rate gain that cancels the unknown steering ratio - the right way to get K_us from these channels; static gains are insensitive to the yaw-estimator filtering. Noisy and only 11.7% coverage, but sound; BEV > ICE (0.79 vs 0.67, AUROC 0.57) as hypothesised. Preferred over steer_understeer_term.
import numpy as np

def feat_yaw_gain_speed_elasticity(w):
    """d ln(yaw-rate gain r/delta) / d ln v over 8-37 m/s: 1 for a neutral car, falls towards -1 with understeer (K_us v^2 >> g L)."""
    v = w["v_mps"]; d = w["steer_angle_deg"]; r0 = w["yaw_rate_radps"]
    r = np.full(r0.shape, np.nan); r[:-2] = r0[2:]
    edges = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 37.0]
    xs = []; ys = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi) & np.isfinite(r) & np.isfinite(d)
        if m.sum() < 50:
            continue
        x = d[m].astype(float); y = r[m].astype(float)
        if np.std(x) < 2.0:
            continue
        xc = x - x.mean(); yc = y - y.mean(); den = float(np.sum(xc * xc))
        if den < 1e-9:
            continue
        s = float(np.sum(xc * yc) / den)
        if not np.isfinite(s) or s <= 1e-5:
            continue
        xs.append(np.log(float(np.mean(v[m])))); ys.append(np.log(s))
    if len(xs) < 2:
        return float("nan")
    xs = np.array(xs); ys = np.array(ys)
    if xs.max() - xs.min() < np.log(1.3):
        return float("nan")
    xc = xs - xs.mean(); yc = ys - ys.mean()
    e = float(np.sum(xc * yc) / np.sum(xc * xc))
    if not np.isfinite(e) or e < -3.0 or e > 3.0:
        return float("nan")
    return e
