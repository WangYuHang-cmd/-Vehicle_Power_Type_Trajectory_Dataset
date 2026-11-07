# Feature stop_nobrake_share (lens: gap_fillers) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: A conventional ICE drivetrain cannot bring the car to a standstill without the friction brake (idle creep torque keeps it rolling), so the brake pedal is pressed in the final second of essentially every stop. BEVs with one-pedal/strong-regen modes (and some HEVs in B-mode, but not to zero) decelerate to a standstill on motor regen alone and hold it electronically, leaving brake_pressed == 0 through the last second. Stops are standstill onsets whose approach traces back (<= 15 s, no intermediate standstill) to v > 2 m/s with das_enabled == 0 throughout; the feature is the share of such stops with brake_pressed == 0 over the final 1.0 s.
# Hypothesis: BEV >> ICE (observed medians 1.0 vs 0.0). Size-neutral in physics; observed rho +0.23 only because BEVs in this sample skew larger.
# Channels: v_mps, a_mps2, brake_pressed, das_enabled, standstill
# Training-sample statistics: finite 0.17, AUROC BEV-vs-ICE 0.771, Spearman(footprint) 0.230
# Audit: keep_with_caveat - Retained representative of the three-way duplicate set (stop_without_brake_frac and brakeless_stop_fraction are 98% identical to it); strictest definition (das == 0 over the whole approach, no intermediate standstill) and highest coverage (16.6%). Physics sound (BEV 1.0 vs ICE 0.0, AUROC 0.77). Caveats: one-pedal/hold-to-stop is a selectable regen setting; brake_pressed derivation differs by port; size relation is the BEV mix (within-ICE rho 0.05).
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

def feat_stop_nobrake_share(w):
    """Share of human stops (from > 2 m/s) reached with brake_pressed == 0 throughout the last 1.0 s (one-pedal / regen-to-stop)."""
    st = _stops(w)
    if not st:
        return _NAN
    return _out(np.mean([s[0] for s in st]))
