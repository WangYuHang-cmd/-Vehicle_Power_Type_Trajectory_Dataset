# Feature ackermann_gain_2_6 (lens: gap_fillers) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: At low speed the understeer term vanishes and kappa / delta_rad ~= 1/(i_s L): path curvature obtained per radian of hand-wheel angle. Median over human-steered samples at 2-6 m/s with |delta| >= 10 deg (both non-zero), >= 10 samples. Shorter wheelbase and quicker steering give more curvature per unit wheel angle, so small cars have a larger gain. The median makes it robust to the yaw-rate lag transients that produce occasional sign disagreements.
# Hypothesis: Decreases with footprint: SMALL > MID > LARGE (observed 0.0241 / 0.0214 / 0.0185 1/m). Powertrain-neutral (observed BEV 0.0211 vs ICE 0.0211, AUROC 0.47).
# Channels: v_mps, das_enabled, curv_1pm, steer_angle_deg
# Training-sample statistics: finite 0.25, AUROC BEV-vs-ICE 0.470, Spearman(footprint) -0.564
# Audit: keep - Robust median estimate of 1/(i_s L) at 2-6 m/s; size rho -0.56 (within-ICE -0.72), powertrain-neutral (0.47), as hypothesised. Redundant with curv_gain_lowspeed (0.90) but a different estimator (ratio median vs OLS slope) with human-only conditioning; keep both and let the classifier choose. das conditioning could be dropped to raise coverage since the gain does not depend on who steers.
import numpy as np
_NAN = float("nan")

_ACK_MIN_N = 10

def _f(w, name):
    return np.asarray(w[name], dtype=np.float64)

def _finite(*arrs):
    m = np.ones(arrs[0].shape, dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m

def _out(x):
    x = float(x)
    return x if np.isfinite(x) else _NAN

def _ackermann(w, vlo, vhi):
    v = _f(w, "v_mps"); das = _f(w, "das_enabled"); k = _f(w, "curv_1pm"); d = _f(w, "steer_angle_deg")
    m = _finite(v, das, k, d) & (v >= vlo) & (v < vhi) & (das == 0) & (np.abs(d) >= 10.0) & (k != 0)
    if m.sum() < _ACK_MIN_N:
        return _NAN
    return _out(np.median(k[m] / np.deg2rad(d[m])))

def feat_ackermann_gain_2_6(w):
    """median curvature per rad of hand-wheel angle at 2-6 m/s (~ 1/(i_s L))."""
    return _ackermann(w, 2.0, 6.0)
