# Feature steer_effort_saturation (lens: gap_fillers) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: EPS assist maps are progressive: assist grows faster than rack force, so driver torque per unit lateral acceleration falls as cornering gets harder. The ratio of the through-origin slope of |torque| vs |lat_acc| above 2 m/s^2 to the slope below 2 m/s^2 measures how strongly the assist curve saturates the driver's effort (1 = linear, < 1 = progressive, > 1 = degressive). Requires >= 40 samples in total and >= 10 in each band, denominator slope > 0.02. Being a ratio it cancels the drive-level torque normalisation.
# Hypothesis: BEVs with strong assist and comfort tuning expected to saturate harder (ratio lower than ICE); observed essentially equal (0.63 vs 0.62), so the hypothesis is not supported in this sample. Size direction weak (observed rho +0.07).
# Channels: v_mps, das_enabled, steer_torque_norm, lat_acc_mps2
# Training-sample statistics: finite 0.17, AUROC BEV-vs-ICE 0.515, Spearman(footprint) 0.072
# Audit: keep - Dimensionless ratio of through-origin slopes that cancels the drive-level normalisation; measures assist-curve progressiveness (plus EPS friction offset at low lateral acceleration, still a vehicle property). Essentially no in-sample separation (AUROC 0.515, size rho 0.07), but the rationale is sound and it is independent of the other torque features (rho 0.03 with steer_effort_slope); the classifier decides its weight.
import numpy as np
_NAN = float("nan")

_TQ_MIN_N = 40
_TQ_MIN_BAND = 10

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

def _tq_mask(w):
    v = _f(w, "v_mps"); das = _f(w, "das_enabled"); tq = _f(w, "steer_torque_norm"); lat = _f(w, "lat_acc_mps2")
    m = _finite(v, das, tq, lat) & (v >= 2.0) & (das == 0) & (np.abs(lat) >= 0.5) & (np.abs(lat) <= 4.0)
    return m, tq, lat

def _slope0(x, y):
    sxx = np.sum(x * x)
    return _NAN if sxx <= 0 else float(np.sum(x * y) / sxx)

def feat_steer_effort_saturation(w):
    """Slope of |torque| vs |lat_acc| above 2 m/s^2 divided by the slope below 2 m/s^2 (assist-curve saturation)."""
    m, tq, lat = _tq_mask(w)
    if m.sum() < _TQ_MIN_N:
        return _NAN
    alat = np.abs(lat); atq = np.abs(tq)
    hi = m & (alat >= 2.0); lo = m & (alat < 2.0)
    if hi.sum() < _TQ_MIN_BAND or lo.sum() < _TQ_MIN_BAND:
        return _NAN
    s_lo = _slope0(alat[lo], atq[lo]); s_hi = _slope0(alat[hi], atq[hi])
    if not (np.isfinite(s_lo) and np.isfinite(s_hi)) or s_lo <= 0.02:
        return _NAN
    return _out(s_hi / s_lo)
