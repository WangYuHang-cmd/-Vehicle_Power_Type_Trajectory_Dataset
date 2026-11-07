# Feature stop_final_decel (lens: gap_fillers) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: The last metre-per-second of a stop reveals the stopping mechanism: regenerative one-pedal stops taper the motor torque smoothly to zero (controller-shaped, gentle), whereas friction-brake stops by a human end with a firmer pedal and the characteristic brake-blending jerk at standstill. Mean of -a_mps2 over samples with 0 < v <= 1 m/s in the human-driven approach, averaged over stops in the window.
# Hypothesis: BEV < ICE (observed 0.67 vs 0.90 m/s^2). Size-neutral (observed rho 0.00).
# Channels: v_mps, a_mps2, brake_pressed, das_enabled, standstill
# Training-sample statistics: finite 0.17, AUROC BEV-vs-ICE 0.342, Spearman(footprint) 0.001
# Audit: keep_with_caveat - Shares the stop detector; plausible regen-taper vs friction-stop mechanism. Caveats: part of the signal is the one-pedal share (rho -0.37 with stop_nobrake_share; within brake-pedal stops the separation shrinks to AUROC 0.445, BEV 0.82 vs ICE 0.92), part is driver brake modulation, and aEgo in the 0-1 m/s band depends on port-specific wheel-speed dead zones. Not a duplicate of stop_decel_taper_ratio (0.53).
import numpy as np
_NAN = float("nan")

_STOP_MAX_LOOKBACK = 150

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

def _stops(w):
    v = _f(w, "v_mps"); a = _f(w, "a_mps2"); brk = _f(w, "brake_pressed"); das = _f(w, "das_enabled"); ss = _f(w, "standstill")
    ok = _finite(v, a, brk, das, ss)
    onsets = np.flatnonzero((ss[1:] == 1) & (ss[:-1] == 0)) + 1
    out = []  # (no_brake_last_1s: bool, final_decel: float or nan)
    for o in onsets:
        j = o - 1
        while j >= 0 and ok[j] and v[j] <= 2.0 and ss[j] == 0 and (o - j) <= _STOP_MAX_LOOKBACK:
            j -= 1
        if j < 0 or not ok[j] or ss[j] == 1 or v[j] <= 2.0:
            continue                       # approach not traced back to v > 2 m/s
        seg = slice(j, o + 1)
        if not np.all(ok[seg]) or np.any(das[seg] != 0):
            continue                       # human-driven approach only
        if o < 10:
            continue
        no_brake = bool(np.all(brk[o - 10:o] == 0))
        fin = (v[j:o] <= 1.0) & (v[j:o] > 0.0)
        final_decel = float(np.mean(-a[j:o][fin])) if fin.sum() >= 2 else _NAN
        out.append((no_brake, final_decel))
    return out

def feat_stop_final_decel(w):
    """Mean deceleration (m/s^2) over the last 1 m/s before standstill, averaged over human stops."""
    st = _stops(w)
    vals = [s[1] for s in st if np.isfinite(s[1])]
    if not vals:
        return _NAN
    return _out(np.mean(vals))
