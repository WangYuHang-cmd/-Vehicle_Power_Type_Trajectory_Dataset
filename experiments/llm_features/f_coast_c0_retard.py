# Feature coast_c0_retard (lens: gap_fillers) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Intercept of the same coast fit: the speed-independent retardation with both pedals released. For ICE this is rolling resistance (Crr g ~ 0.1 m/s^2) plus in-gear engine braking (0.2-0.6 m/s^2); for BEVs it is dominated by default regenerative braking at pedal release (typically 0.7-1.5 m/s^2 in standard/one-pedal modes). So for BEVs this deliberately mixes regen with rolling/drivetrain losses, and that mixing is exactly the discriminating signal; the value is therefore a powertrain-behaviour cue, not a clean rolling-resistance estimate. Grade is unknown and adds symmetric noise (p5 -0.44 m/s^2 are downhill windows).
# Hypothesis: BEV >> ICE (observed 1.15 vs 0.30 m/s^2). Size direction weak/none (rolling resistance per unit mass is size-independent); observed rho -0.15 reflects the LARGE class being ICE-heavy.
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled
# Training-sample statistics: finite 0.12, AUROC BEV-vs-ICE 0.721, Spearman(footprint) -0.145
# Audit: keep_with_caveat - Effectively the pedal-free coast deceleration (BEV 1.15 vs ICE 0.30 m/s^2, AUROC 0.72), which the rationale openly treats as a regen-setting cue. Caveats: regen level is a driver setting; the intercept inherits the noisy c2 term (rho -0.65), so it is not a clean v -> 0 value; nearly redundant with coast_decel_low_speed_mps2 (-0.94) and liftoff_decel_moving (-0.71), which are the cleaner estimators.
import numpy as np
_NAN = float("nan")

_COAST_MIN_N = 30
_COAST_SETTLE = 5
_COAST_MIN_SPREAD = 3.0

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

def _coast_fit(w):
    v = _f(w, "v_mps"); a = _f(w, "a_mps2"); gas = _f(w, "gas_pressed"); brk = _f(w, "brake_pressed"); das = _f(w, "das_enabled")
    c = _finite(v, a, gas, brk, das) & (gas == 0) & (brk == 0) & (das == 0) & (v > 5.0)
    cs = c.copy()
    for lag in range(1, _COAST_SETTLE + 1):
        cs[lag:] &= c[:-lag]
        cs[:lag] = False
    if cs.sum() < _COAST_MIN_N:
        return None
    vv = v[cs]; y = -a[cs]
    if np.percentile(vv, 90) - np.percentile(vv, 10) < _COAST_MIN_SPREAD:
        return None
    A = np.column_stack([np.ones(vv.size), vv * vv])
    scale = np.sqrt(np.sum(A * A, axis=0))
    coef = np.linalg.lstsq(A / scale, y, rcond=None)[0] / scale
    return coef

def feat_coast_c0_retard(w):
    """c0 (m/s^2): speed-independent coast retardation (rolling + drivetrain drag / regen)."""
    c = _coast_fit(w)
    return _NAN if c is None else _out(c[0])
