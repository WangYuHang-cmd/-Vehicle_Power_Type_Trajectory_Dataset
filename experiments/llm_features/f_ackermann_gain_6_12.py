# Feature ackermann_gain_6_12 (lens: gap_fillers) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Same ratio kappa / delta_rad at 6-12 m/s: 1/(i_s L + i_s K_us v^2 / g). It still scales inversely with wheelbase x steering ratio but is reduced by understeer growing with v^2, so the pair (2-6, 6-12) brackets the geometric gain and the understeer build-up without a regression. Human-steered samples only, |delta| >= 10 deg, >= 10 samples.
# Hypothesis: Decreases with footprint (observed 0.0224 / 0.0205 / 0.0184 1/m). Slightly higher for BEV than ICE if BEVs are closer to neutral steer (observed 0.0204 vs 0.0200, AUROC 0.54, weak).
# Channels: v_mps, das_enabled, curv_1pm, steer_angle_deg
# Training-sample statistics: finite 0.25, AUROC BEV-vs-ICE 0.540, Spearman(footprint) -0.400
# Audit: keep - Same ratio at 6-12 m/s where understeer starts to reduce the gain; size rho -0.40 (within-ICE -0.65); weak BEV > ICE (0.54) consistent with more neutral BEV tuning. Moderately correlated with ackermann_gain_2_6 (0.63) so the pair does carry the speed dependence it claims. Sound.
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

def feat_ackermann_gain_6_12(w):
    """median curvature per rad of hand-wheel angle at 6-12 m/s (geometric gain minus understeer)."""
    return _ackermann(w, 6.0, 12.0)
