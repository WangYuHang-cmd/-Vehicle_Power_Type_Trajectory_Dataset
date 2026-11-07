# Feature steer_geom_gain (lens: gap_fillers) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Bicycle model: hand-wheel angle delta = c0 + (i_s L) kappa + (i_s K_us/g) kappa v^2. The coefficient on kappa is the product of wheelbase L and steering ratio i_s, i.e. how much the driver must turn the wheel per unit path curvature at low speed. Longer-wheelbase (larger-footprint) vehicles need more wheel angle for the same curvature, and large SUVs/trucks also use slower steering ratios, so the gain grows with size. Fitted over human-steered moving samples only (das_enabled == 0, v >= 2 m/s, |kappa| >= 1e-3), least squares with intercept, >= 40 samples, condition-number guard.
# Hypothesis: Increases with footprint: SMALL < MID < LARGE (observed medians 40.8 / 45.5 / 53.6 m). No strong powertrain direction expected; BEV skateboard platforms have long wheelbases but often quick steering ratios (observed BEV 46.9 vs ICE 44.7 m, AUROC 0.55).
# Channels: v_mps, das_enabled, curv_1pm, steer_angle_deg
# Training-sample statistics: finite 0.34, AUROC BEV-vs-ICE 0.550, Spearman(footprint) 0.443
# Audit: keep - i_s * L from the two-term bicycle-model fit, well guarded (condition number, robust refit); the medians (47 m) agree with the inverse estimators 1/ackermann_gain_2_6 (47.6 m) and 1/curv_gain_lowspeed, size rho 0.44 (within-ICE 0.56), powertrain-neutral (0.55). Minor: no yaw-lag shift, so curvature noise gives a small attenuation bias; the das == 0 restriction is unnecessary for a transfer gain and only costs coverage. Redundant with the two low-speed gains (|rho| 0.72-0.78) but not a duplicate.
import numpy as np
_NAN = float("nan")

_STEER_MIN_N = 40
_STEER_COND_MAX = 20.0

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

def _steer_fit(w):
    v = _f(w, "v_mps"); das = _f(w, "das_enabled"); k = _f(w, "curv_1pm"); d = _f(w, "steer_angle_deg")
    m = _finite(v, das, k, d) & (v >= 2.0) & (das == 0) & (np.abs(k) >= 1e-3)
    if m.sum() < _STEER_MIN_N:
        return None
    kk = k[m]; vv = v[m]; dd = np.deg2rad(d[m])
    A = np.column_stack([np.ones(kk.size), kk, kk * vv * vv])
    scale = np.sqrt(np.sum(A * A, axis=0))
    if not np.all(scale > 0):
        return None
    As = A / scale
    s = np.linalg.svd(As, compute_uv=False)
    if s[-1] <= 0 or s[0] / s[-1] > _STEER_COND_MAX:
        return None
    coef_s = np.linalg.lstsq(As, dd, rcond=None)[0]
    coef = coef_s / scale
    # one robust re-fit: drop samples with |residual| > 3 * 1.4826 * MAD (yaw-rate spikes, slip transients)
    r = dd - A @ coef
    mad = np.median(np.abs(r - np.median(r)))
    if mad > 0:
        keep = np.abs(r) <= 3.0 * 1.4826 * mad
        if keep.sum() >= _STEER_MIN_N and keep.sum() < keep.size:
            A2 = A[keep]; s2 = np.sqrt(np.sum(A2 * A2, axis=0))
            if np.all(s2 > 0):
                As2 = A2 / s2
                sv2 = np.linalg.svd(As2, compute_uv=False)
                if sv2[-1] > 0 and sv2[0] / sv2[-1] <= _STEER_COND_MAX:
                    coef = np.linalg.lstsq(As2, dd[keep], rcond=None)[0] / s2
    return coef

def feat_steer_geom_gain(w):
    """i_s * L in metres: hand-wheel angle per unit path curvature at v -> 0 (two-term fit)."""
    c = _steer_fit(w)
    return _NAN if c is None else _out(c[1])
