# Feature steer_effort_slope (lens: gap_fillers) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Driver steering torque is the rack force (front-axle lateral load x caster/kingpin trail) minus EPS assist. Plotting normalised driver torque against lateral acceleration over human-steered moving samples (|lat_acc| 0.5-4 m/s^2, >= 40 samples) gives the effort gradient of the assist map in units of the drive's own peak torque: how much of the driver's typical maximum effort each m/s^2 of lateral acceleration costs. Heavy, front-loaded BEVs are tuned with high assist and light on-centre feel, lowering the gradient; many ICE platforms keep more build-up. Least squares through the origin on magnitudes (sign-convention proof).
# Hypothesis: BEV < ICE (observed 0.33 vs 0.45 per m/s^2). Weak positive with footprint since larger vehicles have more rack force to feel (observed rho +0.14; SMALL 0.36, MID 0.42, LARGE 0.41).
# Channels: v_mps, das_enabled, steer_torque_norm, lat_acc_mps2
# Training-sample statistics: finite 0.32, AUROC BEV-vs-ICE 0.306, Spearman(footprint) 0.145
# Audit: keep_with_caveat - Part of the separation is EPS assist shape (AUROC 0.37 after renormalising torque within the window) but part comes through the drive-level torque normaliser (window p95 of normalised torque BEV 0.91 vs ICE 0.99, rho 0.49 with the slope), i.e. the drive's route/manoeuvre composition and the torque signal's tail; the size relation (0.145) disappears after renormalisation (0.07). Caveats: EPS assist tuning is a design property not dynamics, driver hold style enters, and the drive-level normalisation is a non-vehicle component.
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

def feat_steer_effort_slope(w):
    """Least-squares slope through the origin of |normalised steering torque| vs |lateral acceleration| (per m/s^2).
    Magnitudes are used so that a torque sign convention that differs between car ports cannot act as a cue."""
    m, tq, lat = _tq_mask(w)
    if m.sum() < _TQ_MIN_N:
        return _NAN
    return _out(_slope0(np.abs(lat[m]), np.abs(tq[m])))
